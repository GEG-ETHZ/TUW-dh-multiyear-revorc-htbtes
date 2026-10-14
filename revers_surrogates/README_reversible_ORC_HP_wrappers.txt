REVERSIBLE ORC-HP SURROGATE WRAPPERS
====================================

This document gives simple instructions for the design, ORC off-design, and
HP off-design prediction wrappers.


PYTHON DEPENDENCIES
-------------------

- Python 3.10 or newer
- NumPy
- pandas
- joblib
- scikit-learn

Install the required packages with:

python -m pip install numpy pandas joblib scikit-learn

Use a compatible scikit-learn version if loading a model produces a version
warning or deserialization error.


EXPECTED FOLDER LAYOUT
----------------------

integration_folder/
    reversible_design_ann_vopt6_wrapper.py
    design_ann_vopt6_predictor.py
    orc_offdesign_surrogate_filtered_variants_vopt6.py
    hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py
    models/
        design_ann_vopt6_module.joblib
        orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib
        orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib
        hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib

Run the examples from integration_folder.


1. DESIGN WRAPPER
-----------------

Wrapper:

reversible_design_ann_vopt6_wrapper.py

Model:

models/design_ann_vopt6_module.joblib

The design ANN represents a nominal module designed at 10 kg/s. For larger
total brine flows, the wrapper selects an integer number of modules and splits
the available flow equally.

The design convention uses:

T_brine_in_HP_C = T_brine_in_ANN_C - 40 K


Design from a total heating requirement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:

total-heating-load-mw
    Required total heating capacity in MW.

total-brine-flow-kg-s
    Total available brine mass flow in kg/s.

T-brine-in-hp-c
    Brine temperature at the HP evaporator inlet in degrees Celsius.

avg-T-amb-c
    Average ambient temperature in degrees Celsius.

p-brine-bar
    Brine pressure in bar.

T-brine-out-min-orc-c
    Minimum ORC-side brine outlet temperature in degrees Celsius.

Example:

python .\reversible_design_ann_vopt6_wrapper.py hp `
  --total-heating-load-mw 2.2 `
  --total-brine-flow-kg-s 25 `
  --T-brine-in-hp-c 65 `
  --avg-T-amb-c 5 `
  --p-brine-bar 18 `
  --T-brine-out-min-orc-c 70


Design from the well condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inputs:

total-brine-flow-kg-s
    Total available brine mass flow in kg/s.

T-brine-well-c
    Well/ORC brine inlet temperature in degrees Celsius.

avg-T-amb-c
    Average ambient temperature in degrees Celsius.

p-brine-bar
    Brine pressure in bar.

T-brine-out-min-orc-c
    Minimum ORC-side brine outlet temperature in degrees Celsius.

grid-points
    Number of candidate module design targets evaluated. Default: 201.

grid-csv-out
    Optional path for saving the evaluated candidate grid.

Example:

python .\reversible_design_ann_vopt6_wrapper.py orc `
  --total-brine-flow-kg-s 25 `
  --T-brine-well-c 106 `
  --avg-T-amb-c 5 `
  --p-brine-bar 18 `
  --T-brine-out-min-orc-c 70 `
  --grid-points 201


Main design outputs
~~~~~~~~~~~~~~~~~~~

temperature_mapping
    Temperatures supplied to and implied by the design convention.

modularisation
    Module count, nominal module flow, actual equal flow split, and module
    design target.

per_module_design
    HP, ORC, heat-exchanger, ACC, and source design for one module.
    It includes orc_design_pressure_ratio and orc_expander_stage_count. Pass
    either the stage count or the design evaporation/condensation pressures to
    the ORC off-design wrapper.

installed_total_design
    Installation-level extensive quantities scaled by the module count.

inverse_search
    Candidate-search information when the well-condition command is used.

warnings
    Training-boundary or nominal-versus-actual-flow cautions.


2. ORC OFF-DESIGN WRAPPER
-------------------------

Wrapper:

orc_offdesign_surrogate_filtered_variants_vopt6.py

Models:

models/orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib
    Existing one-stage expander model.

models/orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib
    Two-stage expander model.

Inputs:

T-amb-c
    Current ambient temperature in degrees Celsius.

T-brine-in-c
    ORC brine inlet temperature in degrees Celsius.

p-brine-bar
    Brine pressure in bar.

T-brine-out-min-c
    Minimum permitted ORC brine outlet temperature in degrees Celsius.

m-brine-max-kg-s
    Available brine mass flow for one module in kg/s.

variant
    Optional model variant. If omitted, the variant stored in the model bundle
    is used.

design-expander-stage-count
    Optional explicit design topology: 1 or 2.

design-p-evap-bar and design-p-cond-bar
    Design evaporation and condensation pressures. When these are provided,
    the wrapper calculates the design pressure ratio. A ratio greater than
    8.0 selects the two-stage model; otherwise it selects the one-stage model.

Provide either design-expander-stage-count or both design pressures. If no
topology information is supplied, the wrapper keeps the legacy one-stage
behaviour.

One-stage example:

python .\orc_offdesign_surrogate_filtered_variants_vopt6.py `
  --model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib `
  --two-stage-model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib `
  --T-amb-c 15 `
  --T-brine-in-c 106 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 70 `
  --m-brine-max-kg-s 8.33 `
  --design-p-evap-bar 28 `
  --design-p-cond-bar 4

Two-stage example:

python .\orc_offdesign_surrogate_filtered_variants_vopt6.py `
  --model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib `
  --two-stage-model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib `
  --T-amb-c 0 `
  --T-brine-in-c 106 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 70 `
  --m-brine-max-kg-s 8.33 `
  --design-p-evap-bar 28 `
  --design-p-cond-bar 3

Main ORC outputs:

orc_Q_in_W
orc_Q_rej_W
orc_P_gross_W
orc_P_net_W
orc_eta_gross
orc_eta_net
orc_P_pump_el_W
orc_P_acc_fan_W
orc_P_other_aux_W
orc_m_brine_kg_s
orc_T_brine_out_C
orc_Q_geo_avail_W
orc_Q_geo_used_W

The output also includes expander_selection, which reports the selected stage
count, design pressure ratio, threshold, and model path. It also includes the
selected variant, all available variant results, derived inputs, and any
training-domain warnings.


3. HP OFF-DESIGN WRAPPER
------------------------

Wrapper:

hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py

Model:

models/hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib

Recommended operating domain:

HP brine inlet temperature:          55 to 75 degrees Celsius
HP minimum brine outlet temperature: 35 to 45 degrees Celsius
Brine pressure:                      12 to 24 bar
Available brine mass flow:            5 to 12 kg/s
Heating load used for stable fitting: approximately 0.2 to 1.0 MW
Sink supply temperature:             75 to 100 degrees Celsius
Sink temperature difference:         10 to 25 K

Inputs:

T-brine-in-c
    Brine temperature at the HP evaporator inlet in degrees Celsius.

p-brine-bar
    Brine pressure in bar.

T-brine-out-min-c
    Minimum permitted HP brine outlet temperature in degrees Celsius. This is
    a lower constraint, not the normal operating outlet setpoint. When the
    source is sufficient, the model predicts a warmer actual outlet and a
    higher evaporation temperature.

m-brine-max-kg-s
    Available brine mass flow for one module in kg/s.

heating-load-mw
    Requested heating load in MW. The alternative heating-load-W argument
    accepts the same quantity in watts.

temperature-lift-K
    Sink supply temperature minus HP brine inlet temperature:

    temperature_lift_K = T_sink_supply_C - T_brine_in_C

sink-deltaT-K
    Sink supply temperature minus sink return temperature.

variant
    Optional model variant. If omitted, the selected inverse-COP variant stored
    in the model bundle is used.

selected-only
    Optional flag that suppresses results from the non-selected variants.

Example:

python .\hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py `
  --model .\models\hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib `
  --T-brine-in-c 65 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 40 `
  --m-brine-max-kg-s 8.33 `
  --heating-load-mw 0.8 `
  --temperature-lift-K 25 `
  --sink-deltaT-K 15 `
  --selected-only


Main HP outputs:

hp_Q_delivered_W
    Predicted useful heat delivered by the HP.

hp_inverse_COP
    Predicted inverse COP.

hp_COP
    Calculated as 1 / hp_inverse_COP.

hp_W_comp_W
    Calculated as hp_Q_delivered_W multiplied by hp_inverse_COP.

hp_Q_source_W
    Calculated as hp_Q_delivered_W minus hp_W_comp_W.

hp_Q_source_surrogate_W
    Independent source-heat estimate retained for diagnostic comparison.

hp_energy_balance_residual_W
    Energy-balance residual. It should be zero, apart from numerical precision.

hp_unmet_load_W
hp_load_coverage_frac
hp_m_brine_kg_s
hp_T_brine_out_C

The following relationship is enforced in the wrapper:

hp_Q_delivered_W = hp_Q_source_W + hp_W_comp_W


GENERAL INTEGRATION NOTES
-------------------------

- Each off-design call represents one module.
- For a multi-module system, first calculate the equal brine-flow split and
  evaluate one representative module at that split flow.
- Multiply extensive module quantities such as power and heat by the number of
  active modules.
- Do not multiply temperatures, pressures, efficiencies, COP values, or
  fractions by the number of modules.
- Use the installed model filenames shown above unless the wrapper is called
  with an explicit alternative model path.
- The wrappers print JSON that can be captured by another Python program or
  external application.
- The prediction functions can also be imported and called directly from
  Python.
- Predictions outside the stated training ranges require engineering review.
