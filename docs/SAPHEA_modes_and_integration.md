# SAPHEA three-mode calculation & MILP integration plan

This note documents (1) exactly how the **SAPHEA LSGEOHP tool** sizes and costs a
geothermal + heat-pump system across its three modes, and (2) how we will move our
MILP from **capacity-as-input** to **capacity-derived-from-the-resource**, the way
SAPHEA does it.

Source: `Saphea_LSGEOHP-tool.html`
(saphea-project/wiki, commit `425bc16`). All formulas below are transcribed
verbatim from the tool's JavaScript.

---

## 1. Shared inputs

Every mode is driven by the **geothermal resource + network temperatures** — never
by a capacity number:

| Symbol | Tool field | Meaning | Munich value |
|--------|-----------|---------|--------------|
| `ṁ`            | `massFlow`        | brine mass flow            | 81.52 kg/s |
| `T_prod`       | `tempProd`        | wellhead production temp   | 106 °C |
| `T_inject`     | `tempInject`      | min. (geological) injection temp | 45 °C |
| `T_supply`     | `tempDhnSupply`   | DHN supply temp            | ~90 °C |
| `T_return`     | `tempDhnReturn`   | DHN return temp            | ~55 °C |
| `T_inject_woHP`| `tempInjectwoHP`  | brine temp leaving the direct HX (CI only) | `T_return + 2` ≈ 65 °C |
| `cp`           | constant          | brine specific heat        | 4.18 kJ/kg·K |

Heat is always `Q = ṁ · cp · ΔT / 1000` → **MW**.

### COP model (`calculateCopVeryHigh`)

```
COP = 0.44532 · (T_supply + 273.15) / (T_supply − T_source)
```

This is a Carnot fraction (factor **0.44532**), where `T_source` is the evaporator
inlet temperature. **Note:** our model currently uses `SAPHEA_ETA = 0.4212`
(calibrated to COP_std = 4.4 at 90/45 °C). The live tool uses **0.44532** — a ~5.6 %
difference to reconcile.

The tool's `COP` is the **evaporator-side ratio** (`Q_evap / W_el`). The heat-output
("standard") COP is `COP + 1`, because `Q_condenser = Q_evap + W_el`.

---

## 2. The three modes

### Mode 1 — Direct Use (DU) — `prodTemp ≥ supplyTemp`

Brine feeds the DHN directly through a heat exchanger; **no heat pump**.

```
Q_DU = ṁ · cp · (T_prod − T_inject) / 1000          [MW]
feasible iff  T_prod ≥ T_supply
```

Munich: `81.52 · 4.18 · (106 − 55) / 1000 = 17.4 MW`.
**Output: thermal capacity + a feasibility flag. No cost.**

### Mode 2 — Capacity Increase (CI) — add an HP in series

The brine is cooled in two stages: a direct HX (106 → 65 °C) **plus** an HP
evaporator that pulls the brine further down (65 → 45 °C). The HP lifts that low-grade
heat to supply temperature.

```
Q_without_HP = ṁ · cp · (T_prod − T_inject_woHP) / 1000     # direct HX  → 14.0 MW
Q_add        = ṁ · cp · (T_inject_woHP − T_inject) / 1000   # HP evaporator (source) → 6.82 MW
COP          = 0.44532·(T_supply+273.15)/(T_supply − T_inject)
W_electric   = Q_add / COP
Q_add_HP     = Q_add + W_electric        # HP condenser OUTPUT  → ~8.8 MW
Q_total      = Q_without_HP + Q_add_HP   # total to DHN         → ~22.8 MW
```

This is exactly the coupling we implemented in `_build_saphea_coupling`:
`Q_HTHP = Q_evap · (1 + 1/COP_SAPHEA)`.

### Mode 3 — Temperature Increase (TI) — `prodTemp < supplyTemp`

Brine is too cold for direct use, so the **entire** flow goes to the HP evaporator.

```
Q_source   = ṁ · cp · (T_prod − T_inject) / 1000      # full brine → evaporator
COP        = 0.44532·(T_supply+273.15)/(T_supply − T_inject)
W_electric = Q_source / COP
Q_total    = Q_source + W_electric                    # all heat via HP
```

Not active for Munich (`T_prod = 106 °C` is always ≥ supply), but kept for generality.

---

## 3. Cost model (HP only — CI & TI modes)

SAPHEA costs **only the heat pump**, never the wells/HX or boiler.

```
COST_REGRESSION (€/MW of HP, economies of scale):
    1 MW → 1,320,000      3 MW → 910,000
   10 MW →   710,000     20 MW → 510,000      (linear interpolation between points)

specificCost   = interpolateCost(size) · 0.8        # €/MW, 0.8 = ~20% reduction factor
investmentCost = Q_add (CI) | Q_source (TI) · specificCost
```

- `size` = `Q_add_HP` (CI) or `Q_total` (TI) — the **condenser output**.
- Investment is charged on the **net added / source MW**, not the condenser MW.

Munich HTHP (≈ 8.82 MW output): `interpolateCost(8.82)=743.7k → ×0.8 = 595k €/MW`,
investment ≈ **4.28 M€**. (Current Excel input is 460k €/MW ≈ 4.06 M€ — same ballpark
on a total-€ basis; see `MEMORY` notes on the basis discrepancy.)

**Geothermal & gas-boiler CAPEX are NOT in SAPHEA** → keep sourcing them from the DEA
Technology Catalogue.

---

## 4. Current model vs. SAPHEA approach

| | Before | Now (implemented) |
|--|--------|-------------------|
| Geo capacity   | hard-coded `Q_GEO_MAX_DIRECT = 17.4`, `Q_GEO_MAX_CI = 14.0` | ✅ derived from `ṁ, T_prod, T_return, pinch`; hourly via return-temp profile |
| HTHP capacity  | `max_output_MW = 8.82` from Excel | ✅ derived: `Q_evap · (1 + 1/COP)` from the resource |
| COP            | `SAPHEA_ETA = 0.4212` (fixed-temp) | ✅ hourly via supply-temp profile (still η = 0.4212 — see Step 4) |
| HTHP CAPEX     | fixed 460k €/MW from Excel | ⬜ still Excel (Step 3) |
| Geo/boiler CAPEX | from Excel (DEA) | unchanged (DEA) |

---

## 5. Integration steps — status

**✅ Step 1 — Resource parameters in the input file.** *Done.* Added a
`Geothermal Resource` sheet to `Munich_tech_data.xlsx` (brine_mass_flow, brine_cp,
T_prod, T_inject_min, T_return_design, direct_hx_pinch, carnot_eta, T_supply_design),
loaded by `load_geothermal_resource()` in `data_loader.py` into
`extra_params["geo_resource"]`.

**✅ Step 2 — `derive_saphea_capacities()` + hourly profiles.** *Done.* The function
in `run_rd.py` computes `Q_geo_du / Q_geo_ci / Q_evap / Q_evap_ti / Q_hthp_max /
cop_std` from the resource. `main()` overrides the geo & HTHP `max_output_MW` with
these design values before `build()`. Inside `_build_saphea_coupling`, the per-`(d,h)`
geo DU/CI capacities, coupling ratio and COP are derived from the **supply & return
temperature profiles** (`rep_supply`, `rep_return`). The old cross-check is now a
consistency safety-net (installed == derived by construction).

**⬜ Step 3 — HTHP CAPEX from the SAPHEA regression.**
Because the brine is **fixed**, the HTHP design size is fixed → its specific cost is a
**constant** (`interpolateCost(Q_HTHP_max)·0.8`). So no piecewise-linear cost is
needed: compute the €/MW once and write it into the technology dict.
⚠️ If we later let HP size be a free decision variable, the size-dependent specific
cost is **concave** (economies of scale) and would need an SOS2 / piecewise-linear
formulation to stay MILP-compatible. For a fixed resource we avoid that entirely.

**⬜ Step 4 — Reconcile the COP factor.** Decide between 0.4212 (our calibration) and
0.44532 (tool) and apply consistently. Currently `carnot_eta = 0.4212` lives in the
resource sheet, so this is now a one-cell change.

**✅ Step 5 — Geo & boiler CAPEX stay from DEA**, since SAPHEA does not provide them.

---

*Generated as a planning note. Formulas transcribed from `Saphea_LSGEOHP-tool.html`
(saphea-project/wiki @ 425bc16).*
