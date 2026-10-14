"""Storage operation over one year, for a single model run: the webinar figure as a per-run output.

Three panels: storage temperature, charge and discharge power, discharge flow per borehole.
Reads only what a finished run writes (btes_timeseries.xlsx and dispatch_hourly_<year>.csv), so it
works both from inside run_rd.py and on any older run folder.

  python tools/make_btes_run_operation.py <results_dir> [year ...]

With no year given it draws every milestone year that has an hourly dispatch file and a non-zero
storage flow. Network temperatures come from the workbook the run used (env DH_DATA, NET_TEMP_SHIFT_K).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CP_F_KJ = 3.827                      # kJ/(kg K), the 20 % glycol of the field simulation
M_MAX_DEFAULT = 0.30                 # pump ceiling, kg/s per borehole
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def _root(results_dir: Path) -> Path:
    for p in results_dir.resolve().parents:
        if (p / "run_rd.py").exists():
            return p
    return Path.cwd()


def _network_temperatures(results_dir: Path):
    """Hourly supply and return temperature of the run, or (None, None) if unavailable."""
    root = _root(results_dir)
    book = root / "data" / os.environ.get("DH_DATA", "Munich_DH_data.xlsx")
    if not book.exists():
        return None, None
    dh = pd.read_excel(book)
    if "supply_temp" not in dh.columns or "return_temp" not in dh.columns:
        return None, None
    shift = float(os.environ.get("NET_TEMP_SHIFT_K", 0) or 0)
    return (dh.supply_temp.values[:8760] - shift), (dh.return_temp.values[:8760] - shift)


def _n_boreholes(results_dir: Path):
    """Field size of the run: from the run's own metadata, else from the environment."""
    for name in ("computational_params.csv", "meta.csv"):
        f = results_dir / name
        if f.exists():
            d = pd.read_csv(f)
            for _, row in d.iterrows():
                if "borehole" in str(row.iloc[0]).lower():
                    try:
                        return int(float(str(row.iloc[1]).split()[0]))
                    except ValueError:
                        pass
    n = os.environ.get("BTES_DESIGN_N")
    return int(n) if n else None


def plot_run(results_dir, years=None, out_dir=None, n_bh=None, m_max=M_MAX_DEFAULT, title=None):
    results_dir = Path(results_dir)
    ts = results_dir / "btes_timeseries.xlsx"
    if not ts.exists():
        return []
    daily = pd.ExcelFile(ts).parse("daily")
    T_sup, T_ret = _network_temperatures(results_dir)
    n_bh = n_bh or _n_boreholes(results_dir)
    out_dir = Path(out_dir or results_dir)
    written = []

    if years is None:
        years = []
        for f in sorted(results_dir.glob("dispatch_hourly_*.csv")):
            y = int(f.stem.split("_")[-1])
            if y in set(daily.year):
                years.append(y)
    for year in years:
        d = daily[daily.year == year].sort_values("day_of_year")
        if d.empty:
            continue
        h = pd.read_csv(results_dir / f"dispatch_hourly_{year}.csv")
        if "BTES (discharge)" not in h.columns:
            continue
        dis = h["BTES (discharge)"].values[:8760]                      # MW
        ch = -h["BTES (charge)"].values[:8760] if "BTES (charge)" in h.columns else np.zeros(8760)
        if dis.max() <= 0 and ch.max() <= 0:
            continue
        day = np.arange(len(dis)) // 24
        dis_d = pd.Series(dis).groupby(day).max().reindex(range(365)).values
        ch_d = pd.Series(ch).groupby(day).max().reindex(range(365)).values

        flow_d = None
        if T_sup is not None and n_bh:
            span = T_sup - T_ret
            with np.errstate(divide="ignore", invalid="ignore"):
                flow = np.where((dis > 1e-3) & (span > 0), dis * 1e3 / (n_bh * CP_F_KJ * span), np.nan)
            flow_d = pd.Series(flow).groupby(day).max().reindex(range(365)).values

        n_panels = 3 if flow_d is not None else 2
        plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "axes.titlesize": 11,
                             "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                             "axes.spines.top": False, "axes.spines.right": False})
        f, ax = plt.subplots(n_panels, 1, figsize=(14.5, 2.6 * n_panels + 0.6), sharex=True,
                            gridspec_kw=dict(hspace=0.12, left=0.125, right=0.985, top=0.955, bottom=0.085))
        ax = np.atleast_1d(ax)

        ax[0].plot(d.day_of_year.values - 1, d.T_borehole_C.values, color="#c2185b", lw=2)
        tmax = float(d.T_borehole_C.max())
        ax[0].axhline(np.ceil(tmax), color="#7a2e2e", ls="--", lw=1.0, alpha=0.6)
        ax[0].set_ylabel("storage temperature [°C]")
        ax[0].set_title(title if title is not None else
                        (f"{results_dir.name} · {year}" + (f" · {n_bh} boreholes" if n_bh else "")),
                        fontsize=11, loc="left")

        ax[1].axhline(0, color="#444", lw=0.8)
        ax[1].fill_between(np.arange(365), 0, ch_d, color="#0f766e", alpha=0.18, lw=0, step="mid")
        ax[1].step(np.arange(365), ch_d, where="mid", color="#0f766e", lw=1.2)
        ax[1].fill_between(np.arange(365), 0, -dis_d, color="#c2185b", alpha=0.18, lw=0, step="mid")
        ax[1].step(np.arange(365), -dis_d, where="mid", color="#c2185b", lw=1.2)
        ax[1].set_ylabel("power [MW]\n(daily maximum)")
        ax[1].text(0.005, 0.95, "charge ↑", transform=ax[1].transAxes, ha="left", va="top",
                   fontsize=10.5, color="#0f766e", fontweight="bold")
        ax[1].text(0.005, 0.05, "discharge ↓", transform=ax[1].transAxes, ha="left", va="bottom",
                   fontsize=10.5, color="#c2185b", fontweight="bold")
        ax[1].text(0.52, 0.30,
                   f"charge up to {np.nanmax(ch_d):.1f} MW, discharge up to {np.nanmax(dis_d):.1f} MW  ·  "
                   f"{np.nansum(ch)/1e3:.1f} / {np.nansum(dis)/1e3:.1f} GWh per year",
                   transform=ax[1].transAxes, ha="center", va="top", fontsize=10.5, color="#444")

        if flow_d is not None:
            ax[2].plot(np.arange(365), flow_d, color="#c2185b", lw=1.4)
            ax[2].axhline(m_max, color="#555", ls=":", lw=1.2)
            ax[2].text(362, m_max * 1.02, f"pump ceiling {m_max:.2f} kg/s per borehole",
                       color="#555", ha="right", fontsize=10)
            ax[2].set_ylabel("discharge flow [kg/s per bh]\n(daily maximum)")
            ax[2].set_ylim(0, m_max * 1.18)

        ax[-1].set_xlabel(str(year))
        for a in ax:
            a.grid(color="#eee"); a.set_axisbelow(True); a.set_xlim(0, 365)
        ax[-1].set_xticks(CUM); ax[-1].set_xticklabels(MONTHS)
        out = out_dir / f"btes_operation_{year}.png"
        f.savefig(out, dpi=150, facecolor="white"); plt.close(f)
        written.append(out)
    return written


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    yrs = [int(a) for a in sys.argv[2:]] or None
    for p in plot_run(sys.argv[1], yrs):
        print("wrote", p)
