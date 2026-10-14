Emmaboda replay (2026-09-07). Same g-function store model as data/btes/gen/make_btes_design.py (validate_field.py, imposed loads).
Geometry/ground: Nordell, Liuzzo Scorpo, Andersson, Rydell, Carlsson 2016 (LTU report, diva2:997525): 140 bh 10x14, 150 m, 4 m,
open hole 115 mm (r_b 0.0575), bedrock 3.0 W/mK (TRT), 2600 kg/m3 x 900 J/kgK, T_g 8 C; top insulation NOT modelled (surface at T_g).
Loads: Table 3 monthly injection Jul 2010-Jul 2013 (5842 MWh, no extraction); Aug-Dec 2013 scaled so 2013 = 2.5 GWh (report text);
2014-2015 ASSUMED 2.2 GWh in / 0.3 GWh out; 2016/17-2020/21 from Andersson, Rydell, Hakansson 2021 (IEA HPT Annex 52) table:
in 2153/1667/1818/2550/3556, out 310/239/2300/1832/1785 MWh. Measured GT1/GT2 digitised from Fig. 25a/b (+-1.5 K).
Result: Aug 2013 model 29.5-30.1 C vs measured 33.5 (GT1) / 31 (GT2) vs report COMSOL 30; 2014-18 model field mean 30-39 C vs '~45 C' core.
v3 (emmaboda_replay_v3.csv): monthly shapes from Annex 52 Fig. 14; monitoring years Sept-Aug. Measured 2017-21 digitised from Annex 52 Fig. 17
(GT1 117 m, GT2 70 m) and Fig. 15 (fluid leaving the store during extraction). Result: 2017-21 GT1 - model mean +0.1 K (rmse 3.6),
GT2 +2.8 K, fluid exit +0.0 K; seasonal swings 2018/19 16.5 vs 15.1 K, 2019/20 10.0 vs 13.0, 2020/21 15.0 vs 14.1; model minima ~1 month early
(extraction shape ends in March in the model, real extraction runs into April/May).
