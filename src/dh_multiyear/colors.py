"""
Munich case study — technology colors and color lookup.

Color palette inspired by the geothermal/HP system character of Munich South.
"""

import matplotlib.pyplot as plt

TECH_COLORS: dict = {
    # Geothermal direct heat — earth/terracotta
    "Geothermal Heat":               "#C0392B",
    "Geothermal Heat_discharge":     "#C0392B",
    "Geothermal Heat_charge":        "#E74C3C",

    # HTHP — high-temperature heat pump, purple
    "HTHP":                          "#8E44AD",
    "HTHP_discharge":                "#8E44AD",
    "HTHP_charge":                   "#A569BD",

    # Gas Boiler — warm amber
    "Gas Boiler":                    "#E67E22",
    "Gas Boiler_discharge":          "#E67E22",
    "Gas Boiler_charge":             "#F39C12",

    # Electric Boiler — steel blue
    "Electric Boiler":               "#2980B9",
    "Electric Boiler_discharge":     "#2980B9",
    "Electric Boiler_charge":        "#5DADE2",

    # Reversible HP/ORC (FLEXGEO) — teal
    "RevORC HP":                     "#16A085",
    "RevORC HP_discharge":           "#16A085",
    "RevORC HP_charge":              "#48C9B0",

    # Unmet demand / curtailment
    "Unmet demand":                  "#BDC3C7",
    "curtailment":                   "#BDC3C7",
}


# Storage flows: every store FAMILY gets its own colour pair (discharge darker,
# charge lighter), so seasonal pit, buffer tank and BTES are distinguishable in
# every plot. Matched by substring of the series name.
STORAGE_FAMILY_COLORS = {
    #                (discharge,  charge)
    "seasonal": ("#27AE60", "#A9DFBF"),   # pit / seasonal store — green pair
    "buffer":   ("#2E86C1", "#AED6F1"),   # short-term buffer    — blue pair
    "btes":     ("#C2185B", "#F48FB1"),   # HT-BTES              — magenta pair
}
# fallback for stores that match no family
STORAGE_DISCHARGE_COLOR = "#7F8C8D"
STORAGE_CHARGE_COLOR    = "#BDC3C7"


def get_color(name: str, i: int = 0) -> object:
    """Return the designated color for *name*, or a sensible fallback.

    Storage series are detected by a ``(discharge)`` / ``(charge)`` suffix;
    the store family (seasonal / buffer / btes) selects the colour pair.
    """
    if name in TECH_COLORS:
        return TECH_COLORS[name]
    low = name.lower().strip()
    if low.endswith("(discharge)") or low.endswith("(charge)"):
        is_dis = low.endswith("(discharge)")
        for key, (dis, ch) in STORAGE_FAMILY_COLORS.items():
            if key in low:
                return dis if is_dis else ch
        return STORAGE_DISCHARGE_COLOR if is_dis else STORAGE_CHARGE_COLOR
    return plt.cm.tab10(abs(hash(name)) % 10)
