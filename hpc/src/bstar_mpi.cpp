// bstar_mpi.cpp -- MPI task farm for critical-benefit estimates (design doc v5, section 3.B.5).
//
// One task = one estimate of b* (network, rule, eps1, eps2, theta, delta, seed, target precision, initial guess).
// Rank 0 hands out tasks one at a time (largest estimated cost first); every other rank runs tasks single-threaded
// and appends one line per finished task to <outdir>/rank_NNNN.txt.  On start, all rank_*.txt files in <outdir> are
// scanned and finished task ids are skipped, so an interrupted job is resumed by simply resubmitting it.
//
// Usage:   mpiexec -n P ./bstar_mpi <tasks.txt> <nets_dir> <outdir>
// Serial build (no MPI, for local tests):  compile with -DNO_MPI;  ./bstar_mpi <tasks.txt> <nets_dir> <outdir>
//
// Task file: '#' comment lines, then whitespace-separated columns
//     id net rule eps1 eps2 theta delta seed target guess cost
// Output line (tab-separated):
//     id net rule eps1 eps2 theta delta seed bstar se runs_per_point calls flag seconds
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifndef NO_MPI
#include <mpi.h>
#endif
#ifdef _WIN32
#include <filesystem>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <cerrno>
#endif
#include "model.hpp"

struct Task {
    int64_t id; std::string net, rule; double eps1, eps2, theta, delta; int64_t seed; double target, guess, cost;
};

static std::vector<Task> read_tasks(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open task file " + path);
    std::vector<Task> v;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        Task t;
        if (is >> t.id >> t.net >> t.rule >> t.eps1 >> t.eps2 >> t.theta >> t.delta >> t.seed >> t.target >> t.guess >> t.cost)
            v.push_back(t);
        else
            throw std::runtime_error("bad task line in " + path + ": " + line);   // never skip a task silently
    }
    std::set<int64_t> ids;
    for (auto& t : v)
        if (!ids.insert(t.id).second) throw std::runtime_error("duplicate task id " + std::to_string(t.id) + " in " + path);
    return v;
}

static std::vector<std::string> list_rank_files(const std::string& dir) {
    std::vector<std::string> out;
#ifdef _WIN32
    if (!std::filesystem::exists(dir)) return out;
    for (auto& e : std::filesystem::directory_iterator(dir)) {
        std::string fn = e.path().filename().string();
        if (fn.rfind("rank_", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".txt") out.push_back(e.path().string());
    }
#else
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        std::string fn = e->d_name;
        if (fn.rfind("rank_", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".txt") out.push_back(dir + "/" + fn);
    }
    closedir(d);
#endif
    return out;
}

static void make_dir(const std::string& dir) {
#ifdef _WIN32
    std::filesystem::create_directories(dir);
#else
    std::string cur;                                                          // mkdir -p without spawning a shell
    for (size_t p = 0; p <= dir.size(); ++p) {
        if (p == dir.size() || dir[p] == '/') {
            if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                throw std::runtime_error("cannot create directory " + cur);
        }
        if (p < dir.size()) cur += dir[p];
    }
#endif
}

// open the rank file for appending; a line left unfinished by a killed job is terminated first, so that it can
// never be glued to the next result (incomplete lines have fewer than 14 fields and are ignored when reading)
static std::ofstream open_rank_file(const std::string& path) {
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open " + path + " for writing");
    out << '\n' << std::flush;
    return out;
}

// ids of tasks that already have a complete output line (14 fields)
static std::set<int64_t> finished_ids(const std::string& dir) {
    std::set<int64_t> done;
    for (auto& fn : list_rank_files(dir)) {
        std::ifstream f(fn);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            std::vector<std::string> tok;
            std::string x;
            while (is >> x) tok.push_back(x);
            if (tok.size() == 14) done.insert(std::stoll(tok[0]));
        }
    }
    return done;
}

// run cap of the refine step (environment variable BSTAR_RCAP, total runs per point; default 1.6e6 as in E1-E3a).
// The estimate is accepted at the first doubling with at least this many runs: 1.6e6 -> 2.56e6, 5.12e6 -> 5.12e6,
// 1.024e7 -> 1.024e7.  Tasks that reach the target precision earlier are not affected.
static int64_t run_cap() {
    const char* s = std::getenv("BSTAR_RCAP");
    if (!s || !*s) return 1600000;
    long long v = std::atoll(s);
    if (v < 40000) throw std::runtime_error(std::string("bad BSTAR_RCAP: ") + s);
    return (int64_t)v;
}

// ------------------------------------------------------------------------------------------------ checkpoints
// 2026-10-08: <outdir>/ckpt/task_<id>.txt holds the refine state at the start of the latest round of an unfinished
// task.  A resubmitted job continues from there (any rank, any mode) and gives the same result as an uninterrupted
// run.  Written to a temporary file and renamed, so a killed job never leaves half a checkpoint.
static std::string ckpt_path(const std::string& outdir, int64_t id) {
    return outdir + "/ckpt/task_" + std::to_string(id) + ".txt";
}
static std::string task_sig(const Task& t) {
    std::ostringstream os;
    os << std::setprecision(17) << t.id << ' ' << t.net << ' ' << t.rule << ' ' << t.eps1 << ' ' << t.eps2 << ' '
       << t.theta << ' ' << t.delta << ' ' << t.seed << ' ' << t.target << ' ' << run_cap();
    return os.str();
}
static void save_ckpt(const std::string& path, const Task& t, const model::Snapshot& s) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        if (!f) return;                                                     // a checkpoint is optional
        f << "ckpt1 " << task_sig(t) << "\n" << std::setprecision(17) << s.calls << ' ' << s.attempt << ' ' << s.b0
          << ' ' << s.h << ' ' << s.runs << ' ' << s.R << ' ' << s.fc.size() << "\n";
        for (double x : s.fc) f << x << ' ';
        f << "\n";
        for (double x : s.fd) f << x << ' ';
        f << "\nend\n";
    }
#ifdef _WIN32
    std::remove(path.c_str());
#endif
    std::rename(tmp.c_str(), path.c_str());
}
static model::Snapshot load_ckpt(const std::string& path, const Task& t) {
    model::Snapshot s;
    std::ifstream f(path);
    if (!f) return s;
    std::string head, sig;
    f >> head;
    std::getline(f, sig);
    size_t p0 = sig.find_first_not_of(' ');
    if (head != "ckpt1" || p0 == std::string::npos || sig.substr(p0) != task_sig(t)) return s;  // other task or cap
    size_t n = 0;
    if (!(f >> s.calls >> s.attempt >> s.b0 >> s.h >> s.runs >> s.R >> n)) return s;
    s.fc.resize(n); s.fd.resize(n);
    for (auto& x : s.fc) f >> x;
    for (auto& x : s.fd) f >> x;
    std::string end;
    f >> end;
    s.valid = (bool)f && end == "end";
    return s;
}

// how one task is run: who writes checkpoints, and (wide mode) which share of the runs this process does
struct Ctx {
    std::string outdir;
    bool writer = true;
    int64_t ishare = 0, nshare = 1;
    std::function<void(std::vector<int64_t>&)> reduce;
    std::function<void()> sync;          // wide mode: all ranks have read the checkpoint before rank 0 may rewrite it
};

static std::string run_task(const Task& t, const std::string& nets_dir, std::map<std::string, model::Net>& cache,
                            const Ctx& cx) {
    auto it = cache.find(t.net);
    if (it == cache.end()) {
        if (cache.size() > 4) cache.clear();
        it = cache.emplace(t.net, model::load_net(nets_dir + "/" + t.net + ".txt")).first;
    }
    auto t0 = std::chrono::steady_clock::now();
    model::Estimator est(it->second, t.rule, t.eps1, t.eps2, t.theta, t.delta, t.seed);
    est.set_rcap(run_cap());
    est.set_share(cx.ishare, cx.nshare, cx.reduce);
    const std::string cp = ckpt_path(cx.outdir, t.id);
    model::Snapshot snap = load_ckpt(cp, t);
    if (cx.sync) cx.sync();
    if (snap.valid && cx.writer)
        std::cout << "task " << t.id << ": resumed from checkpoint (window " << snap.attempt << ", " << snap.runs
                  << " runs per batch done)\n" << std::flush;
    std::function<void(const model::Snapshot&)> save;
    if (cx.writer) save = [&cp, &t](const model::Snapshot& s) { save_ckpt(cp, t, s); };
    est.set_checkpoint(save, &snap);
    model::Result r = est.estimate(t.guess, t.target);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ostringstream os;
    os << t.id << '\t' << t.net << '\t' << t.rule << '\t' << t.eps1 << '\t' << t.eps2 << '\t' << t.theta << '\t' << t.delta
       << '\t' << t.seed << '\t' << std::setprecision(10) << r.bstar << '\t' << r.se << '\t' << r.runs_per_point << '\t'
       << r.calls << '\t' << (r.flag.empty() ? "ok" : r.flag) << '\t' << std::setprecision(6) << secs;
    return os.str();
}

int main(int argc, char** argv) {
    int rank = 0, size = 1;
#ifndef NO_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    if (argc < 4) {
        if (rank == 0) std::cerr << "usage: bstar_mpi <tasks.txt> <nets_dir> <outdir>\n";
#ifndef NO_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    const std::string task_file = argv[1], nets_dir = argv[2], outdir = argv[3];
    std::vector<Task> tasks = read_tasks(task_file);
    std::ostringstream fn;
    fn << outdir << "/rank_" << std::setw(4) << std::setfill('0') << rank << ".txt";
    std::map<std::string, model::Net> cache;

    const bool wide = std::getenv("BSTAR_WIDE") && std::string(std::getenv("BSTAR_WIDE")) == "1";
    (void)wide;                                                               // unused in the serial build
    std::vector<size_t> order(tasks.size());                                  // largest estimated cost first
    for (size_t q = 0; q < order.size(); ++q) order[q] = q;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return tasks[a].cost > tasks[b].cost; });

    if (size == 1) {                                                          // serial mode
        make_dir(outdir);
        make_dir(outdir + "/ckpt");
        std::set<int64_t> done = finished_ids(outdir);
        std::ofstream out = open_rank_file(fn.str());
        size_t todo = 0;
        for (size_t q : order) if (!done.count(tasks[q].id)) ++todo;
        // BSTAR_TESTSHARE=n (local test only): every call is done as n separate shares whose counts are summed,
        // exactly as n processes of the wide mode would do it
        Ctx cx; cx.outdir = outdir;
        if (const char* ts = std::getenv("BSTAR_TESTSHARE")) cx.nshare = std::max<int64_t>(1, std::atoll(ts));
        std::cout << "serial mode: " << tasks.size() << " tasks, " << done.size() << " already done, " << todo
                  << " to run; run cap " << run_cap() << (cx.nshare > 1 ? "; test shares " + std::to_string(cx.nshare) : "")
                  << "\n" << std::flush;
        for (size_t q : order) {
            if (done.count(tasks[q].id)) continue;
            std::string line = run_task(tasks[q], nets_dir, cache, cx);
            out << line << "\n" << std::flush;
            std::remove(ckpt_path(outdir, tasks[q].id).c_str());
            std::cout << line << "\n" << std::flush;
        }
#ifndef NO_MPI
        MPI_Finalize();
#endif
        return 0;
    }

#ifndef NO_MPI
    if (wide) {                                                               // wide mode: all ranks on one task
        // 2026-10-08: for tasks too long for one 30-min node (theta = 0.1 at large run caps).  Every rank does a
        // share of every call; the counts are summed with MPI_Allreduce, so all ranks follow the same steps and get
        // the same (bit-identical) result.  Tasks are done one after the other; rank 0 writes rank_0000.txt and the
        // checkpoints.
        if (rank == 0) { make_dir(outdir); make_dir(outdir + "/ckpt"); }
        MPI_Barrier(MPI_COMM_WORLD);
        std::set<int64_t> done = finished_ids(outdir);
        MPI_Barrier(MPI_COMM_WORLD);
        std::ofstream out;
        if (rank == 0) out = open_rank_file(fn.str());
        size_t todo = 0;
        for (size_t q : order) if (!done.count(tasks[q].id)) ++todo;
        if (rank == 0)
            std::cout << "wide mode: " << tasks.size() << " tasks, " << done.size() << " already done, " << todo
                      << " to run, each on all " << size << " ranks; run cap " << run_cap() << "\n" << std::flush;
        Ctx cx; cx.outdir = outdir; cx.writer = (rank == 0); cx.ishare = rank; cx.nshare = size;
        cx.sync = []() { MPI_Barrier(MPI_COMM_WORLD); };
        cx.reduce = [](std::vector<int64_t>& c) {
            MPI_Allreduce(MPI_IN_PLACE, c.data(), (int)c.size(), MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
        };
        double t0 = MPI_Wtime();
        for (size_t q : order) {
            if (done.count(tasks[q].id)) continue;
            MPI_Barrier(MPI_COMM_WORLD);                                      // checkpoint file complete before reads
            std::string line = run_task(tasks[q], nets_dir, cache, cx);
            if (rank == 0) {
                out << line << "\n" << std::flush;
                std::remove(ckpt_path(outdir, tasks[q].id).c_str());
                std::cout << line << "   [" << (MPI_Wtime() - t0) / 60.0 << " min]\n" << std::flush;
            }
        }
        MPI_Finalize();
        return 0;
    }
#endif

#ifndef NO_MPI
    const int TAG_REQ = 1, TAG_TASK = 2;
    if (rank == 0) {
        make_dir(outdir);
        std::set<int64_t> done = finished_ids(outdir);
        make_dir(outdir + "/ckpt");
        std::vector<int64_t> queue;                                           // indices into tasks, largest cost first
        for (size_t q : order) if (!done.count(tasks[q].id)) queue.push_back((int64_t)q);
        std::cout << "master: " << tasks.size() << " tasks, " << done.size() << " already done, " << queue.size()
                  << " to run on " << size - 1 << " workers; run cap " << run_cap() << "\n" << std::flush;
        MPI_Barrier(MPI_COMM_WORLD);
        size_t next = 0;
        int active = size - 1;
        double t0 = MPI_Wtime();
        while (active > 0) {
            int64_t dummy;
            MPI_Status st;
            MPI_Recv(&dummy, 1, MPI_INT64_T, MPI_ANY_SOURCE, TAG_REQ, MPI_COMM_WORLD, &st);
            int64_t give = -1;
            if (next < queue.size()) give = queue[next++];
            else --active;
            MPI_Send(&give, 1, MPI_INT64_T, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
            if (give >= 0 && (next % 50 == 0 || next == queue.size()))
                std::cout << "master: dispatched " << next << " / " << queue.size() << " after " << (MPI_Wtime() - t0) / 60.0
                          << " min\n" << std::flush;
        }
        std::cout << "master: all workers finished after " << (MPI_Wtime() - t0) / 60.0 << " min\n" << std::flush;
    } else {
        MPI_Barrier(MPI_COMM_WORLD);                                          // outdir exists after the barrier
        std::ofstream out;
        try { out = open_rank_file(fn.str()); }
        catch (const std::exception& ex) { std::cerr << "rank " << rank << ": " << ex.what() << "\n"; MPI_Abort(MPI_COMM_WORLD, 2); }
        Ctx cx; cx.outdir = outdir;
        while (true) {
            int64_t req = rank, give;
            MPI_Send(&req, 1, MPI_INT64_T, 0, TAG_REQ, MPI_COMM_WORLD);
            MPI_Recv(&give, 1, MPI_INT64_T, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (give < 0) break;
            std::string line = run_task(tasks[(size_t)give], nets_dir, cache, cx);
            out << line << "\n" << std::flush;
            std::remove(ckpt_path(outdir, tasks[(size_t)give].id).c_str());
            std::cout << "rank " << rank << ": " << line << "\n" << std::flush;
        }
    }
    MPI_Finalize();
#endif
    return 0;
}
