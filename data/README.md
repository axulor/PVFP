# Networks

All 40 networks are simple, undirected and connected (largest connected component, nodes relabelled 0..N−1). They are built by `code/networks40.py` (`build(name)`); the exact edge lists used in the simulations are in `hpc/nets/`.

Statistics: N players, E edges, mean degree ⟨k⟩, coefficient of variation of degree CV, mean embeddedness ⟨e⟩, relative embeddedness η and triangle degree disparity d̄ (definitions in Section 3 of the paper).

## Synthetic networks

N = 500 before taking the largest component; one instance per network with a fixed seed (see `code/networks40.py`). In the figures, HK and RHG networks are labelled by their two parameters, e.g. HK 4/0.9 and RHG 3.0/0.1.

| Network | Model | Parameters | N | E | ⟨k⟩ | CV | ⟨e⟩ | η | d̄ |
|---|---|---|---|---|---|---|---|---|---|
| ER_k4 | Erdős–Rényi G(N, M) | mean degree k = 4 | 491 | 1000 | 4.07 | 0.46 | 0.04 | 0.01 | 1.54 |
| ER_k8 | Erdős–Rényi G(N, M) | k = 8 | 499 | 2000 | 8.02 | 0.36 | 0.13 | 0.01 | 1.72 |
| ER_k16 | Erdős–Rényi G(N, M) | k = 16 | 500 | 4000 | 16.00 | 0.25 | 0.51 | 0.03 | 1.52 |
| BA_m2 | Barabási–Albert | m = 2 edges per new node | 500 | 996 | 3.98 | 1.22 | 0.14 | 0.01 | 11.90 |
| BA_m4 | Barabási–Albert | m = 4 | 500 | 1984 | 7.94 | 0.95 | 0.55 | 0.03 | 6.87 |
| BA_m8 | Barabási–Albert | m = 8 | 500 | 3936 | 15.74 | 0.84 | 1.94 | 0.05 | 5.16 |
| CL_g2.2 | Chung–Lu, power-law expected degrees | exponent γ = 2.2, mean degree 8 | 496 | 1945 | 7.84 | 1.10 | 1.13 | 0.03 | 4.34 |
| CL_g2.6 | Chung–Lu | γ = 2.6 | 493 | 1949 | 7.91 | 0.81 | 0.43 | 0.02 | 3.85 |
| CL_g3.2 | Chung–Lu | γ = 3.2 | 499 | 1951 | 7.82 | 0.59 | 0.21 | 0.02 | 2.60 |
| WS_p0.05 | Watts–Strogatz (connected), k = 8 | rewiring probability p = 0.05 | 500 | 2000 | 8.00 | 0.08 | 3.77 | 0.54 | 1.12 |
| WS_p0.2 | Watts–Strogatz | p = 0.2 | 500 | 2000 | 8.00 | 0.15 | 2.31 | 0.33 | 1.27 |
| HK_m4_p0.3 | Holme–Kim | m = 4, triad-formation probability p = 0.3 | 500 | 1983 | 7.93 | 1.04 | 1.15 | 0.09 | 7.17 |
| HK_m4_p0.6 | Holme–Kim | m = 4, p = 0.6 | 500 | 1979 | 7.92 | 1.11 | 1.92 | 0.18 | 8.13 |
| HK_m4_p0.9 | Holme–Kim | m = 4, p = 0.9 | 500 | 1983 | 7.93 | 1.08 | 2.75 | 0.27 | 7.40 |
| HK_m8_p0.9 | Holme–Kim | m = 8, p = 0.9 | 500 | 3928 | 15.71 | 0.92 | 4.60 | 0.19 | 5.76 |
| RGG_k6 | random geometric graph on the unit torus | mean degree k = 6 | 493 | 1495 | 6.06 | 0.37 | 3.48 | 0.56 | 1.39 |
| RGG_k12 | random geometric graph | k = 12 | 500 | 3020 | 12.08 | 0.30 | 7.16 | 0.57 | 1.32 |
| RHG_g3.0_T0.1 | random hyperbolic graph, mean degree 8 | exponent γ = 3.0, temperature T = 0.1 | 490 | 1996 | 8.15 | 0.99 | 5.81 | 0.55 | 4.33 |
| RHG_g3.0_T0.5 | random hyperbolic graph | γ = 3.0, T = 0.5 | 498 | 2001 | 8.04 | 1.02 | 3.21 | 0.26 | 4.90 |
| RHG_g2.7_T0.1 | random hyperbolic graph | γ = 2.7, T = 0.1 | 494 | 2000 | 8.10 | 1.64 | 5.89 | 0.42 | 9.85 |

## Real social networks

| Network | Description and construction | Reference | Source | N | E | ⟨k⟩ | CV | ⟨e⟩ | η | d̄ |
|---|---|---|---|---|---|---|---|---|---|---|
| CPH | Copenhagen Networks Study, Bluetooth proximity of students; edge if ≥ 15 five-minute windows with RSSI ≥ −75 dBm | Sapiezynski et al., Sci. Data 6, 315 (2019) | https://doi.org/10.6084/m9.figshare.7267433 (bt_symmetric.csv) | 648 | 4575 | 14.12 | 0.65 | 7.58 | 0.36 | 2.01 |
| UTM | Utah middle school, one day of sensor contacts (19.5 s windows); edge if ≥ 15 windows | Leecaster et al., PLoS ONE 11, e0153690 (2016) | https://doi.org/10.1371/journal.pone.0153690 (file S2) | 629 | 5515 | 17.54 | 0.40 | 2.84 | 0.13 | 1.75 |
| UTE | Utah elementary school (Elem1), two days, 2 m criterion (19.5 s windows); edge if ≥ 15 windows | Toth et al., J. R. Soc. Interface 12, 20150279 (2015) | https://doi.org/10.1098/rsif.2015.0279 (data D3, D4) | 322 | 3017 | 18.74 | 0.26 | 11.87 | 0.63 | 1.33 |
| SFHH | SFHH conference, face-to-face contacts (20 s windows); edge if ≥ 3 windows | Génois & Barrat, EPJ Data Sci. 7, 11 (2018) | https://sociopatterns.org/datasets/sfhh-conference-data-set/ | 401 | 3196 | 15.94 | 0.85 | 4.45 | 0.16 | 3.54 |
| LPS | Lyon primary school, face-to-face contacts (20 s windows); edge if ≥ 15 windows | Stehlé et al., PLoS ONE 6, e23176 (2011) | https://networks.skewed.de/net/sp_primary_school | 242 | 1938 | 16.02 | 0.44 | 8.47 | 0.44 | 1.68 |
| THS | French high school (Thiers 2013), proximity (20 s windows); edge if ≥ 15 windows | Mastrandrea et al., PLoS ONE 10, e0136497 (2015) | https://networks.skewed.de/net/sp_high_school (proximity) | 324 | 1435 | 8.86 | 0.54 | 3.65 | 0.32 | 2.01 |
| LH10 | Lyon hospital ward, face-to-face contacts (20 s windows); edge if ≥ 15 windows | Vanhems et al., PLoS ONE 8, e73970 (2013) | https://networks.skewed.de/net/sp_hospital | 71 | 391 | 11.01 | 0.81 | 7.73 | 0.30 | 2.45 |
| KAR | Zachary karate club | Zachary, J. Anthropol. Res. 33, 452 (1977) | networkx.karate_club_graph() | 34 | 78 | 4.59 | 0.83 | 1.73 | 0.29 | 4.35 |
| ESH | Spanish high school t6, mutual positive nominations | Ruiz-García et al., PNAS 120, e2215041120 (2023) | https://doi.org/10.5281/zenodo.7647000 (Edges_t6.csv) | 475 | 2996 | 12.61 | 0.42 | 6.52 | 0.47 | 1.71 |
| RH | residence hall friendships; edge if either names the other | Freeman et al., Soc. Networks 20, 109 (1998) | https://networks.skewed.de/net/residence_hall | 217 | 1839 | 16.95 | 0.46 | 5.92 | 0.31 | 1.97 |
| UGF | Ugandan village 8, household friendships; edge if either names the other | Chami et al., PNAS 114, E7425 (2017) | https://networks.skewed.de/net/ugandan_village (friendship-8) | 369 | 1753 | 9.50 | 0.70 | 0.74 | 0.05 | 4.24 |
| CPH-FB | Copenhagen Networks Study, Facebook friendships | Sapiezynski et al., Sci. Data 6, 315 (2019) | https://doi.org/10.6084/m9.figshare.7267433 (fb_friends.csv) | 800 | 6418 | 16.05 | 0.83 | 6.40 | 0.23 | 2.80 |
| FBF | Facebook ego network | Maier & Brockmann, Phys. Rev. E 96, 042307 (2017) | https://networks.skewed.de/net/facebook_friends | 329 | 1954 | 11.88 | 0.92 | 10.73 | 0.43 | 2.39 |
| EEU | email-Eu, two people sent or received the same email; edge if ≥ 15 common emails | Benson et al., PNAS 115, E11221 (2018) | https://www.cs.cornell.edu/~arb/data/email-Eu/ | 719 | 7114 | 19.79 | 0.86 | 14.19 | 0.38 | 2.32 |
| URV | URV university email network | Guimerà et al., Phys. Rev. E 68, 065103 (2003) | http://deim.urv.cat/~alexandre.arenas/data/xarxes/ (email.zip) | 1133 | 5451 | 9.62 | 0.97 | 2.94 | 0.14 | 2.83 |
| JAZZ | jazz musicians playing in the same band (projection) | Gleiser & Danon, Adv. Complex Syst. 6, 565 (2003) | https://networks.skewed.de/net/jazz_collab | 198 | 2742 | 27.70 | 0.63 | 19.58 | 0.48 | 2.19 |
| NS | network-science coauthorships (projection) | Newman, Phys. Rev. E 74, 036104 (2006) | https://networks.skewed.de/net/netscience | 379 | 914 | 4.82 | 0.81 | 3.02 | 0.53 | 3.10 |
| DOL | dolphin associations | Lusseau et al., Behav. Ecol. Sociobiol. 54, 396 (2003) | https://networks.skewed.de/net/dolphins | 62 | 159 | 5.13 | 0.57 | 1.79 | 0.23 | 1.74 |
| UGH | Ugandan village 9, household health advice; edge if either names the other | Chami et al., PNAS 114, E7425 (2017) | https://networks.skewed.de/net/ugandan_village (health-advice_9) | 173 | 623 | 7.20 | 1.42 | 1.97 | 0.11 | 12.64 |
| TER | 2004 Madrid train bombing, associations among terrorists | Hayes, Am. Sci. 94, 400 (2006) | https://networks.skewed.de/net/train_terrorists | 64 | 243 | 7.59 | 0.81 | 6.51 | 0.48 | 2.30 |

## Files

- `nets/*.npz`: CPH, CPH-FB, UTM, UTE, SFHH, EEU, URV, ESH and KAR, processed from the raw downloads (edge list `edges`, largest component).
- `socio/*.npz`: edge weights (number of 20 s contact windows) of LPS, THS and LH10.
- `netz/*.csv.zip`: the Netzschleuder files used for RH, UGF, UGH, FBF, JAZZ, NS, DOL and TER.

The raw downloads are not included. To rebuild `nets/*.npz`, download the files below from the sources in the table, place them under `data/raw/`, run `python build_nets.py real` in `code/`, and rename `ES-t6.npz` to `ESH.npz`.

- `copenhagen/bt_symmetric.csv`, `copenhagen/fb_friends.csv`
- `utah_middle_school/pone.0153690.s002.csv`
- `utah_schools_toth2015/rsif20150279supp4.txt`, `rsif20150279supp5.txt` (data D3 and D4)
- `sociopatterns_SFHH/SFHH_tij.dat.gz`
- `benson_email-Eu/email-Eu.tar.gz`
- `urv_email/email.zip`
- `spanish_highschools/Edges_t6.csv`
