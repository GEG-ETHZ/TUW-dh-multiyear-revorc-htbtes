"""Replay a monitored BTES with the same g-function store model the surrogate is built on.

Two modes:
  imposed  - drive the field with a given daily net load (W, +charge / -discharge) and return
             the mean borehole-wall temperature; compare with measured store temperatures.
  control  - run our own charge/discharge rule (fixed inlet on charge, UA-limited direct
             discharge to a return temperature, outlet >= setpoint) with the site's temperatures.

Geometry and ground properties are arguments so any published field can be replayed
(Drake Landing, Emmaboda, Braedstrup, ...). Ground is conduction-only, uniform, no groundwater;
top insulation enters only through the surface boundary of the g-function (T = T_g).
Usage example:
  python validate_field.py --name DLSC --n 144 --H 35 --B 2.25 --layout rect --rb 0.0762 \
     --lambda 1.7 --rho-cp 2.5e6 --tg 6 --loads dlsc_daily_loads.csv --out dlsc_replay.csv
loads csv: columns day (0..), Q_W (net, +charge/-discharge). Optional --T-meas csv (day, T_C).
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, pandas as pd
import pygfunction as gt
from scipy.interpolate import interp1d
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
import make_btes_design as mb   # noqa: E402  (disc_field, pipe constants)


def field(n, H, B, layout, rb, D=1.0):
    if layout == "disc":
        return mb.disc_field(n, B, H, D=D, r_b=rb)
    k = int(np.ceil(np.sqrt(n)))
    return gt.boreholes.rectangle_field(k, int(np.ceil(n / k)), B, B, H, D, rb)[:n]


def step_response(boreholes, lam, rho_cp, n_days):
    dt = 86400.0
    t_log = np.geomspace(dt, n_days * dt, 200)
    g = gt.gfunction.uniform_temperature(boreholes, t_log, lam / rho_cp, nSegments=1, disp=False)
    return interp1d(t_log, g, kind="linear", fill_value="extrapolate")(np.arange(1, n_days + 1) * dt) / (2 * np.pi * lam)


def replay(boreholes, lam, rho_cp, T_g, Q_W):
    """Mean wall temperature for an imposed daily load series (W, +charge)."""
    L = len(boreholes) * boreholes[0].H
    n = len(Q_W); h = step_response(boreholes, lam, rho_cp, n)
    q = np.asarray(Q_W, float) / L
    dq = np.diff(np.concatenate(([0.0], q)))
    T = np.full(n + 1, T_g)
    for d in range(n):
        T[d + 1] = T_g + float(np.dot(dq[:d + 1], h[d::-1][:d + 1]))
    return T[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--H", type=float, required=True); ap.add_argument("--B", type=float, required=True)
    ap.add_argument("--layout", default="rect"); ap.add_argument("--rb", type=float, default=0.075)
    ap.add_argument("--D", type=float, default=1.0, help="burial depth of the borehole top, m")
    ap.add_argument("--lambda", dest="lam", type=float, required=True)
    ap.add_argument("--rho-cp", type=float, required=True); ap.add_argument("--tg", type=float, required=True)
    ap.add_argument("--loads", required=True, help="csv with day, Q_W (net, +charge)")
    ap.add_argument("--T-meas", default=None, help="csv with day, T_C measured mean store temperature")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bh = field(a.n, a.H, a.B, a.layout, a.rb, a.D)
    ld = pd.read_csv(a.loads); Q = ld.Q_W.values
    T = replay(bh, a.lam, a.rho_cp, a.tg, Q)
    out = pd.DataFrame({"day": np.arange(len(Q)), "Q_W": Q, "T_wall_C": T})
    print(f"{a.name}: {len(bh)} bh x {a.H} m, B {a.B} m, lambda {a.lam}, rho_cp {a.rho_cp:.2e}, T_g {a.tg}")
    yrs = len(Q) // 365
    for y in range(yrs):
        s = slice(y * 365, (y + 1) * 365); qi = Q[s].clip(min=0).sum() * 24 / 1e6; qo = -Q[s].clip(max=0).sum() * 24 / 1e6
        print(f"  year {y+1}: in {qi:7.0f} MWh  out {qo:7.0f} MWh  ratio {100*qo/qi if qi else 0:5.1f} %  T_wall mean {T[s].mean():5.1f}  max {T[s].max():5.1f}  end {T[s][-1]:5.1f} C")
    if a.T_meas:
        m = pd.read_csv(a.T_meas); mi = np.interp(m.day, out.day, out.T_wall_C)
        out_m = pd.DataFrame({"day": m.day, "T_meas": m.T_C, "T_model": mi}); print("  vs measured:"); print(out_m.round(1).to_string(index=False))
        print(f"  RMSE {np.sqrt(np.mean((mi - m.T_C.values)**2)):.1f} K, bias {np.mean(mi - m.T_C.values):+.1f} K")
        out_m.to_csv(Path(a.out).with_name(Path(a.out).stem + "_vs_measured.csv"), index=False)
    out.to_csv(a.out, index=False); print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
