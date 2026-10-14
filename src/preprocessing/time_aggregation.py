"""
Time Series Aggregation

Utilities for creating representative periods (typical days) from full time series data
using tsam (time series aggregation module).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from loguru import logger

try:
    import tsam.timeseriesaggregation as tsam
    TSAM_AVAILABLE = True
except ImportError:
    logger.warning("tsam not installed. Representative period creation will not be available.")
    TSAM_AVAILABLE = False

# ── Seasonal window definitions (mirrors _day_season in btes_constraints.py) ──
# d is 0-indexed (day-of-year = d + 1)
_CHARGE_DAYS    = [d for d in range(365) if 121 <= (d + 1) <= 273]   # May–Sep (153 days)
_IDLE_DAYS      = [d for d in range(365) if 91 <= (d + 1) <= 120 or 274 <= (d + 1) <= 304]  # April + October (61 days)
_DISCHARGE_DAYS = [d for d in range(365) if (d + 1) <= 90 or (d + 1) >= 305]  # Nov–Mar (151 days)
# NOTE (2026-09-08): October moved from the discharge to the idle window so that these
# windows are identical to _day_season in btes_constraints.py; before, October days sat in
# discharge clusters and 14 of 365 days were handled under a season not their own.

# Ordered list used by create_representative_days_stratified to assign global rep indices:
# discharge first, then idle, then charge (keeps global 0-based index stable)
_SEASON_GROUPS = [
    ("discharge", _DISCHARGE_DAYS),
    ("idle",      _IDLE_DAYS),
    ("charge",    _CHARGE_DAYS),
]


def _proportional_split(n_total: int) -> Dict[str, int]:
    """Allocate n_total rep-days across seasons proportional to their length."""
    if n_total < 3:
        raise TimeAggregationError(
            f"n_periods={n_total} is too small for stratified TSA; need at least 3 "
            "(one representative day per seasonal window)."
        )
    n_dis = max(1, round(151 / 365 * n_total))
    n_ch  = max(1, round(153 / 365 * n_total))
    n_id  = max(1, n_total - n_dis - n_ch)
    # absorb rounding error in discharge (largest window)
    diff  = n_total - (n_dis + n_ch + n_id)
    n_dis = max(1, n_dis + diff)
    return {"discharge": n_dis, "idle": n_id, "charge": n_ch}


class TimeAggregationError(Exception):
    """Raised when time aggregation fails."""
    pass


def create_representative_periods(
    time_series_data: pd.DataFrame,
    n_periods: int = 24,
    hours_per_period: int = 24,
    method: str = "hierarchical",
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create representative periods from full time series data using tsam.
    
    Parameters
    ----------
    time_series_data : pd.DataFrame
        Time series data with datetime index and columns for different variables
        (e.g., heat_demand, temperature, solar_radiation)
    n_periods : int, default=12
        Number of representative periods (typical days) to create
    hours_per_period : int, default=24
        Hours per representative period (24 for daily aggregation)
    method : str, default="hierarchical"
        Clustering method: "hierarchical", "k_means", or "k_medoids"
    **kwargs
        Additional arguments passed to tsam
        
    Returns
    -------
    representative_data : pd.DataFrame
        Aggregated time series for representative periods
    period_weights : pd.Series
        Number of original periods each representative period represents
        
    Raises
    ------
    TimeAggregationError
        If tsam is not available or aggregation fails
    """
    if not TSAM_AVAILABLE:
        raise TimeAggregationError(
            "tsam is not installed. Install it with: pip install tsam"
        )
    
    if len(time_series_data) % hours_per_period != 0:
        raise TimeAggregationError(
            f"Time series length ({len(time_series_data)}) must be divisible by "
            f"hours_per_period ({hours_per_period})"
        )
    
    logger.info(
        f"Creating {n_periods} representative periods from "
        f"{len(time_series_data)} time steps using {method} clustering"
    )
    
    try:
        # Initialize tsam aggregation
        aggregation = tsam.TimeSeriesAggregation(
            time_series_data,
            noTypicalPeriods=n_periods,
            hoursPerPeriod=hours_per_period,
            clusterMethod=method,
            **kwargs
        )
        
        # Perform clustering
        aggregation.createTypicalPeriods()
        
        # Get representative data and weights
        representative_data = aggregation.typicalPeriods
        period_weights = aggregation.clusterPeriodNoOccur
        
        logger.info(
            f"Successfully created {len(representative_data)} representative periods. "
            f"Compression ratio: {len(time_series_data)/len(representative_data):.1f}x"
        )
        
        return representative_data, period_weights
        
    except Exception as e:
        raise TimeAggregationError(f"Failed to create representative periods: {e}")


def create_representative_days_with_mapping(
    time_series_data: pd.DataFrame,
    n_periods: int = 24,
    hours_per_period: int = 24,
    method: str = "hierarchical",
    **kwargs,
) -> Dict:
    """
    Create representative days with the full day-to-rep-day mapping.

    This is needed for the Kotzur inter/intra-day superposition storage
    method (arXiv 2512.00892, Section A.1).  In addition to the
    representative profiles and weights, it returns the *sequence*
    mapping ``day_to_rep`` that tells us which representative day each
    of the 365 original days maps to.

    Parameters
    ----------
    time_series_data : pd.DataFrame
        8760-row DataFrame with datetime index.
    n_periods : int
        Number of representative (typical) days.
    hours_per_period : int
        Hours per period (24 for daily aggregation).
    method : str
        Clustering method passed to tsam.

    Returns
    -------
    dict with keys:
        ``rep_data``      – pd.DataFrame of representative profiles
        ``weights``       – pd.Series of period occurrence counts
        ``day_to_rep``    – list[int] of length 365, mapping original
                            day index → representative day index
        ``n_days``        – int, number of original days (365)
        ``n_rep_days``    – int, number of representative days
    """
    if not TSAM_AVAILABLE:
        raise TimeAggregationError("tsam is not installed.")

    n_total = len(time_series_data)
    if n_total % hours_per_period != 0:
        raise TimeAggregationError(
            f"Time series length ({n_total}) not divisible by "
            f"hours_per_period ({hours_per_period})"
        )

    # Keep the peak-demand day as its own cluster centre. Without it the model's
    # per-cluster averaging clips the annual peak (~8 % here) and undersizes the
    # peak plant; 'replace_cluster_center' does NOT help, because only tsam's
    # centroid profile changes and this model re-averages the member days.
    if kwargs.pop("extreme_periods", True) and "heat_demand" in time_series_data:
        kwargs.setdefault("extremePeriodMethod", "new_cluster_center")
        kwargs.setdefault("addPeakMax", ["heat_demand"])

    aggregation = tsam.TimeSeriesAggregation(
        time_series_data,
        noTypicalPeriods=n_periods,
        hoursPerPeriod=hours_per_period,
        clusterMethod=method,
        **kwargs,
    )
    aggregation.createTypicalPeriods()

    rep_data = aggregation.typicalPeriods  # (n_periods*24) rows
    weights_raw = aggregation.clusterPeriodNoOccur  # dict: cluster_label → count

    # Normalise weights to a dict with int keys if needed
    if isinstance(weights_raw, dict):
        weights = weights_raw
    else:
        weights = dict(weights_raw)

    # --- build day_to_rep mapping ---
    # tsam stores cluster assignments in clusterOrder (numpy array,
    # length = number of original periods = 365)
    cluster_order = list(aggregation.clusterOrder)  # convert to plain list

    # Map cluster labels to contiguous 0-based representative-day indices.
    cluster_labels = sorted(weights.keys())
    label_to_idx = {label: idx for idx, label in enumerate(cluster_labels)}
    day_to_rep = [label_to_idx[label] for label in cluster_order]

    # Convert weights to ordered list matching 0-based rep-day indices
    weight_list = [weights[label] for label in cluster_labels]

    n_days = len(cluster_order)
    # Derive the count from the clustering itself: with extremePeriodMethod
    # tsam ADDS periods (peak day kept as its own centre), so the result can
    # exceed n_periods. Taking n_periods on faith would leave day_to_rep
    # pointing past the end of the model's REP_DAYS set.
    n_rep_days = len(cluster_labels)
    if n_rep_days != n_periods:
        logger.info(
            f"TSA: {n_rep_days} representative days returned for n_periods="
            f"{n_periods} (extreme periods retained)")

    logger.info(
        f"TSA: {n_days} days → {n_rep_days} representative days "
        f"({method} clustering). Weights sum = {sum(weight_list)}."
    )

    return {
        "rep_data": rep_data,
        "weights": weight_list,       # list[int], indexed 0..n_rep-1
        "day_to_rep": day_to_rep,      # list[int], length 365
        "n_days": n_days,
        "n_rep_days": n_rep_days,
    }


def create_representative_days_stratified(
    time_series_data: pd.DataFrame,
    n_periods: int = 12,
    n_periods_per_season: Optional[Dict[str, int]] = None,
    hours_per_period: int = 24,
    method: str = "hierarchical",
    extreme_periods: bool = True,
    peak_max_cols: Optional[list] = None,
    **kwargs,
) -> Dict:
    """
    Season-stratified representative days for the RD (Kotzur) model.

    Runs independent TSA within each of the three BTES seasonal windows:
      - discharge : Nov–Mar (days 1–90 + 305–365, 151 days)
      - idle      : April + October (days 91–120 + 274–304, 61 days)
      - charge    : May–Sep (days 121–273, 153 days)

    Because every representative day belongs to exactly one season, the
    majority-vote season label in ``add_btes_to_rd_model`` is always exact
    and the Kotzur ARX never propagates charge/discharge in the wrong season.

    Parameters
    ----------
    time_series_data : pd.DataFrame
        8760-row DataFrame with a DatetimeIndex and any number of columns.
    n_periods : int
        Total number of representative days.  Must be ≥ 3.
        Split proportionally across seasons unless ``n_periods_per_season``
        is given.
    n_periods_per_season : dict or None
        Explicit split, e.g. ``{"discharge": 6, "idle": 1, "charge": 5}``.
        Values must sum to ``n_periods``.
    hours_per_period : int
        Must be 24 (daily aggregation).
    method : str
        tsam clustering method (``"hierarchical"``, ``"k_means"``, …).
    extreme_periods : bool, default True
        If True, pin the peak-demand day inside the discharge season as a
        mandatory representative day (``extremePeriodMethod='new_cluster_center'``
        in tsam).  This guarantees the coldest/highest-demand day is never
        averaged away by clustering.
    peak_max_cols : list or None
        Columns for which the maximum-value day should be pinned.  Defaults to
        ``['heat_demand']`` if that column is present.
    **kwargs
        Forwarded to ``tsam.TimeSeriesAggregation`` (e.g. ``weightDict``).

    Returns
    -------
    dict  — same keys as ``create_representative_days_with_mapping``:
        ``rep_data``   : pd.DataFrame, shape (n_periods * 24, n_cols)
        ``weights``    : list[int] of length n_periods (occurrence counts)
        ``day_to_rep`` : list[int] of length 365
        ``n_days``     : 365
        ``n_rep_days`` : n_periods
    """
    if not TSAM_AVAILABLE:
        raise TimeAggregationError("tsam is not installed.")
    if hours_per_period != 24:
        raise TimeAggregationError("Stratified mode requires hours_per_period=24.")
    if len(time_series_data) != 8760:
        raise TimeAggregationError(
            f"Expected 8760 rows, got {len(time_series_data)}."
        )

    n_per: Dict[str, int] = n_periods_per_season or _proportional_split(n_periods)
    if sum(n_per.values()) != n_periods:
        raise TimeAggregationError(
            f"n_periods_per_season sums to {sum(n_per.values())}, "
            f"expected {n_periods}."
        )

    # Resolve which columns to pin as extreme-max in the discharge season
    extreme_periods = bool(kwargs.pop("extreme_periods", extreme_periods))
    _peak_cols = peak_max_cols if peak_max_cols is not None else ["heat_demand"]
    _peak_cols = [c for c in _peak_cols if c in time_series_data.columns]

    day_to_rep: list = [0] * 365
    all_weights: list = []
    all_rep_data: list = []
    global_offset = 0

    for season, days_in_season in _SEASON_GROUPS:
        n_s = n_per[season]

        # Extract rows for this season as a contiguous block
        row_indices = [h for d in days_in_season for h in range(d * 24, d * 24 + 24)]
        subset = time_series_data.iloc[row_indices].copy()
        # tsam requires a contiguous DatetimeIndex at hourly frequency
        subset.index = pd.date_range("2025-01-01", periods=len(subset), freq="h")

        # Pin the peak-demand day in the discharge season so the coldest/highest
        # demand day is never smoothed away by the cluster centroid averaging.
        season_kwargs = dict(kwargs)
        if extreme_periods and season == "discharge" and _peak_cols and n_s > 1:
            # 'new_cluster_center', NOT 'replace_cluster_center'. The latter only
            # swaps tsam's centroid profile, which this model never reads: the
            # MILP re-derives demand by averaging the member days of each cluster,
            # so a pinned centroid is discarded and the peak is clipped ~8 %
            # anyway. 'new_cluster_center' puts the peak day in a cluster of its
            # own, so that average IS the peak day and the peak survives exactly.
            season_kwargs["extremePeriodMethod"] = "new_cluster_center"
            season_kwargs["addPeakMax"] = _peak_cols
            logger.info(
                f"  Extreme period: max({_peak_cols}) day kept as its own "
                f"cluster centre in the discharge season"
            )

        agg = tsam.TimeSeriesAggregation(
            subset,
            noTypicalPeriods=n_s,
            hoursPerPeriod=24,
            clusterMethod=method,
            **season_kwargs,
        )
        agg.createTypicalPeriods()

        weights_raw   = dict(agg.clusterPeriodNoOccur)
        cluster_labels = sorted(weights_raw.keys())
        label_to_idx  = {lbl: i for i, lbl in enumerate(cluster_labels)}
        cluster_order = list(agg.clusterOrder)  # length = len(days_in_season)

        for i_local, d in enumerate(days_in_season):
            day_to_rep[d] = global_offset + label_to_idx[cluster_order[i_local]]

        all_weights.extend(int(weights_raw[lbl]) for lbl in cluster_labels)
        all_rep_data.append(agg.typicalPeriods)
        # retaining an extreme period ADDS a cluster, so advance by what the
        # clustering actually returned rather than by the requested n_s
        global_offset += len(cluster_labels)

    combined_rep_data = pd.concat(all_rep_data, ignore_index=True)
    n_rep_total = global_offset
    extreme_tag = f", extreme peak pinned ({_peak_cols})" if extreme_periods and _peak_cols else ""
    logger.info(
        f"Stratified TSA: 365 days → {n_rep_total} rep-days "
        f"(requested {n_periods}: discharge={n_per['discharge']}, "
        f"idle={n_per['idle']}, charge={n_per['charge']})"
        f"{extreme_tag}. Weights sum={sum(all_weights)}."
    )

    return {
        "rep_data":   combined_rep_data,
        "weights":    all_weights,
        "day_to_rep": day_to_rep,
        "n_days":     365,
        "n_rep_days": n_rep_total,
    }


# BTES seasonal windows (0-indexed hours), identical to btes_constraints.py
_RH_CH    = (2880, 6551)          # May 1 - Sep 30   charging
_RH_IDLE2 = (6552, 7295)          # October          idle
_RH_DIS1  = 2159                  # Jan 1 - Mar 31   discharging
_RH_DIS2  = 7296                  # Nov 1 - Dec 31   discharging


def _rh_season(h: int) -> str:
    if _RH_CH[0] <= h <= _RH_CH[1]:
        return "charge"
    if _RH_IDLE2[0] <= h <= _RH_IDLE2[1]:
        return "idle"
    if h <= _RH_DIS1 or h >= _RH_DIS2:
        return "discharge"
    return "idle"


def _cluster_season_pure(time_series_data, T, n_rep_hours, method, columns,
                         segmented=False, **kwargs):
    """Cluster inside each seasonal window separately and concatenate.

    Guarantees that every representative hour belongs to exactly one BTES
    season, so the majority-vote classification in add_btes_to_rh_model()
    is exact rather than approximate. Representative hours are split across
    the windows in proportion to the number of real hours each contains.
    """
    seasons: Dict[str, List[int]] = {}
    for h in range(T):
        seasons.setdefault(_rh_season(h), []).append(h)

    # proportional split, at least one rep hour per non-empty window
    order = [s for s in ("discharge", "charge", "idle") if seasons.get(s)]
    raw = {s: n_rep_hours * len(seasons[s]) / T for s in order}
    alloc = {s: max(1, int(round(raw[s]))) for s in order}
    while sum(alloc.values()) > n_rep_hours:
        alloc[max(order, key=lambda s: alloc[s])] -= 1
    while sum(alloc.values()) < n_rep_hours:
        alloc[max(order, key=lambda s: raw[s] - alloc[s])] += 1

    sigma = [0] * T
    weight_list: List[int] = []
    rep_profiles: Dict[int, Dict[str, float]] = {}
    offset = 0
    for s in order:
        hours = seasons[s]
        sub = time_series_data.iloc[hours].copy()
        sub.index = pd.date_range("2025-01-01", periods=len(sub), freq="h")
        n_s = min(alloc[s], len(hours))
        if segmented:
            sig_s, w_s, prof_s = _cluster_segmented(sub, len(hours), n_s, columns)
        else:
            sig_s, w_s, prof_s = _cluster_independent(
                sub, len(hours), n_s, method, columns, **kwargs)
        for local, h in enumerate(hours):
            sigma[h] = offset + sig_s[local]
        weight_list.extend(w_s)
        # rep_profiles is a dict keyed by rep-hour index in both clusterers,
        # so shift the sub-clustering's keys into the global numbering
        for k, v in prof_s.items():
            rep_profiles[offset + k] = v
        logger.info(f"  RH season '{s}': {len(hours)} h -> {n_s} rep hours")
        offset += n_s

    return sigma, weight_list, rep_profiles


def create_representative_hours_with_mapping(
    time_series_data: pd.DataFrame,
    n_rep_hours: int = 300,
    method: str = "hierarchical",
    segmented: bool = False,
    season_pure: bool = False,
    peak_retention: bool = True,
    **kwargs,
) -> Dict:
    """
    Create representative hours (RH) with the reduced storage index J.

    Implements the RH aggregation from arXiv (Mannhardt et al.) Section 3.3.
    The mapping σ maps each of the T original hours to one of I
    representative hours.

    The *reduced storage index* J is computed by scanning the σ sequence
    and grouping adjacent hours that share the same representative hour
    into single storage-level intervals.

    Parameters
    ----------
    time_series_data : pd.DataFrame
        8760-row DataFrame with datetime index.
    n_rep_hours : int
        Number of representative hours I.
    method : str
        Clustering method (used only when ``segmented=False``).
    segmented : bool, default True
        If True, use connectivity-constrained agglomerative clustering
        (Ward linkage with a sparse tridiagonal adjacency matrix) so
        that only temporally adjacent hours can be merged.  This
        guarantees contiguous segments and produces |J| = I.
        If False, cluster individual hours independently (the original
        approach), which typically gives |J| ≈ T.

    Returns
    -------
    dict with keys:
        ``rep_profiles``     – dict {i: {col: value}} for each rep hour
        ``weights``          – list[int] of length I (occurrence counts)
        ``sigma``            – list[int] of length T (hour→rep_hour mapping)
        ``storage_intervals``– list of dicts with keys:
                               ``rep_hour`` (int), ``duration`` (int),
                               ``start`` (int), ``end`` (int)
        ``n_hours``          – int (8760)
        ``n_rep_hours``      – int (I)
        ``n_storage_levels`` – int (|J|)
    """
    T = len(time_series_data)
    columns = list(time_series_data.columns)

    if season_pure:
        # Cluster each BTES seasonal window on its own, so no representative
        # hour can ever straddle charge / idle / discharge. Without this the
        # model classifies a mixed rep hour by majority vote and the minority
        # hours inherit the wrong season: measured on the Munich series,
        # 53 % of rep hours span a boundary and 12.3 % of the year is
        # misassigned under free clustering (0.3 % when segmented).
        sigma, weight_list, rep_profiles = _cluster_season_pure(
            time_series_data, T, n_rep_hours, method, columns,
            segmented=segmented, **kwargs,
        )
        label = ("segmented Ward" if segmented else method) + ", season-pure"
    elif segmented:
        sigma, weight_list, rep_profiles = _cluster_segmented(
            time_series_data, T, n_rep_hours, columns,
        )
        label = "segmented Ward"
    else:
        if not TSAM_AVAILABLE:
            raise TimeAggregationError("tsam is not installed.")
        sigma, weight_list, rep_profiles = _cluster_independent(
            time_series_data, T, n_rep_hours, method, columns, **kwargs,
        )
        label = method

    # ── Peak retention ───────────────────────────────────────────────
    # Cluster centres are averages, so the single highest-demand hour is
    # otherwise blended with milder hours and the annual peak is clipped
    # (measured on Munich: -2.3 % free, -3.2 % segmented), which under-sizes
    # the peak plant. Give that hour a representative hour of its own. The
    # RD model does the same thing via tsam's extremePeriodMethod.
    if peak_retention and "heat_demand" in time_series_data.columns:
        dem = time_series_data["heat_demand"].to_numpy()
        h_peak = int(np.argmax(dem))
        old_i = sigma[h_peak]
        if weight_list[old_i] > 1:          # only if it shares its cluster
            new_i = len(weight_list)
            sigma[h_peak] = new_i
            weight_list[old_i] -= 1
            weight_list.append(1)
            rep_profiles[new_i] = {c: float(time_series_data[c].iloc[h_peak])
                                   for c in columns}
            # the vacated cluster's centre is the mean of what is left
            members = [t for t, i in enumerate(sigma) if i == old_i]
            if members:
                rep_profiles[old_i] = {
                    c: float(time_series_data[c].to_numpy()[members].mean())
                    for c in columns}
            n_rep_hours += 1
            logger.info(
                f"  Peak hour {h_peak} ({dem[h_peak]:.2f}) kept as its own "
                f"representative hour ({n_rep_hours} total)")

    # ── Build reduced storage index J ────────────────────────────────
    storage_intervals = []
    j = 0
    start = 0
    current_rep = sigma[0]
    for t in range(1, T):
        if sigma[t] != current_rep:
            storage_intervals.append({
                "j": j,
                "rep_hour": current_rep,
                "duration": t - start,
                "start": start,
                "end": t - 1,
            })
            j += 1
            start = t
            current_rep = sigma[t]
    storage_intervals.append({
        "j": j,
        "rep_hour": current_rep,
        "duration": T - start,
        "start": start,
        "end": T - 1,
    })

    n_J = len(storage_intervals)

    logger.info(
        f"RH-TSA: {T} hours → {n_rep_hours} representative hours "
        f"({label} clustering). |J| = {n_J} storage levels. "
        f"Weights sum = {sum(weight_list)}."
    )

    return {
        "rep_profiles": rep_profiles,
        "weights": weight_list,
        "sigma": sigma,
        "storage_intervals": storage_intervals,
        "n_hours": T,
        "n_rep_hours": n_rep_hours,
        "n_storage_levels": n_J,
    }


def _cluster_segmented(
    time_series_data: pd.DataFrame,
    T: int,
    n_rep_hours: int,
    columns: list,
) -> tuple:
    """Connectivity-constrained Ward clustering → contiguous segments."""
    from scipy.sparse import eye as speye
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import MinMaxScaler

    data_raw = time_series_data.values
    data_norm = MinMaxScaler().fit_transform(data_raw)

    adjacency = speye(T, k=1, format="csr") + speye(T, k=-1, format="csr")

    clustering = AgglomerativeClustering(
        n_clusters=n_rep_hours,
        linkage="ward",
        connectivity=adjacency,
    )
    raw_labels = clustering.fit_predict(data_norm)

    # Remap labels to temporal order (label 0 = earliest segment)
    first_occurrence = {}
    for t, lab in enumerate(raw_labels):
        if lab not in first_occurrence:
            first_occurrence[lab] = t
    sorted_labels = sorted(first_occurrence, key=first_occurrence.get)
    remap = {old: new for new, old in enumerate(sorted_labels)}

    sigma = [remap[lab] for lab in raw_labels]

    # Representative profiles = mean of original (unscaled) values per cluster
    weight_list = [0] * n_rep_hours
    accum = {i: np.zeros(len(columns)) for i in range(n_rep_hours)}
    for t in range(T):
        i = sigma[t]
        accum[i] += data_raw[t]
        weight_list[i] += 1

    rep_profiles = {}
    for i in range(n_rep_hours):
        vals = accum[i] / weight_list[i] if weight_list[i] > 0 else accum[i]
        rep_profiles[i] = {col: float(vals[j]) for j, col in enumerate(columns)}

    return sigma, weight_list, rep_profiles


def _cluster_independent(
    time_series_data: pd.DataFrame,
    T: int,
    n_rep_hours: int,
    method: str,
    columns: list,
    **kwargs,
) -> tuple:
    """Original approach: cluster individual hours without temporal order."""
    aggregation = tsam.TimeSeriesAggregation(
        time_series_data,
        noTypicalPeriods=n_rep_hours,
        hoursPerPeriod=1,
        clusterMethod=method,
        **kwargs,
    )
    aggregation.createTypicalPeriods()

    rep_data = aggregation.typicalPeriods
    weights_raw = aggregation.clusterPeriodNoOccur
    weights_dict = dict(weights_raw) if not isinstance(weights_raw, dict) else weights_raw

    cluster_order = list(aggregation.clusterOrder)
    cluster_labels = sorted(weights_dict.keys())
    label_to_idx = {label: idx for idx, label in enumerate(cluster_labels)}
    sigma = [label_to_idx[label] for label in cluster_order]
    weight_list = [weights_dict[label] for label in cluster_labels]

    rep_profiles = {}
    for idx, label in enumerate(cluster_labels):
        row = rep_data.iloc[idx]
        rep_profiles[idx] = {col: float(row[col]) for col in columns}

    return sigma, weight_list, rep_profiles



def create_representative_periods_per_year(
    time_series_data: pd.DataFrame,
    years: list[int],
    n_periods_per_year: int = 12,
    hours_per_period: int = 24,
    method: str = "hierarchical"
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.Series]]:
    """
    Create representative periods for each year separately.
    
    This is useful for multi-year models where each year should have its own
    representative periods (e.g., due to changing demand patterns).
    
    Parameters
    ----------
    time_series_data : pd.DataFrame
        Time series data with datetime index
    years : list of int
        Years to create representative periods for
    n_periods_per_year : int, default=12
        Number of representative periods per year
    hours_per_period : int, default=24
        Hours per representative period
    method : str, default="hierarchical"
        Clustering method
        
    Returns
    -------
    representative_data_dict : dict
        Dictionary mapping year to representative data
    weights_dict : dict
        Dictionary mapping year to period weights
    """
    representative_data_dict = {}
    weights_dict = {}
    
    for year in years:
        # Extract data for this year
        year_data = time_series_data[time_series_data.index.year == year]
        
        if len(year_data) == 0:
            logger.warning(f"No data found for year {year}")
            continue
        
        # Create representative periods for this year
        rep_data, weights = create_representative_periods(
            year_data,
            n_periods=n_periods_per_year,
            hours_per_period=hours_per_period,
            method=method
        )
        
        representative_data_dict[year] = rep_data
        weights_dict[year] = weights
        
        logger.info(f"Created {n_periods_per_year} representative periods for year {year}")
    
    return representative_data_dict, weights_dict


def calculate_period_durations(
    period_weights: pd.Series,
    hours_per_period: int = 24
) -> pd.Series:
    """
    Calculate total duration represented by each typical period in hours.
    
    Parameters
    ----------
    period_weights : pd.Series
        Number of original periods each representative period represents
    hours_per_period : int, default=24
        Hours per period
        
    Returns
    -------
    pd.Series
        Total hours represented by each period
    """
    return period_weights * hours_per_period


def validate_representative_periods(
    original_data: pd.DataFrame,
    representative_data: pd.DataFrame,
    period_weights: pd.Series
) -> Dict[str, float]:
    """
    Validate representative periods by comparing statistics with original data.
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original time series
    representative_data : pd.DataFrame
        Representative periods
    period_weights : pd.Series
        Period weights
        
    Returns
    -------
    dict
        Validation statistics for each variable
    """
    validation_stats = {}
    
    for column in original_data.columns:
        original_mean = original_data[column].mean()
        
        # Calculate weighted mean of representative periods
        rep_mean = (representative_data[column] * period_weights.values.repeat(
            len(representative_data) // len(period_weights)
        )).sum() / period_weights.sum() / (len(representative_data) // len(period_weights))
        
        relative_error = abs(rep_mean - original_mean) / original_mean * 100
        
        validation_stats[column] = {
            'original_mean': original_mean,
            'representative_mean': rep_mean,
            'relative_error_pct': relative_error
        }
        
        logger.info(
            f"{column}: Original mean={original_mean:.2f}, "
            f"Representative mean={rep_mean:.2f}, "
            f"Error={relative_error:.2f}%"
        )
    
    return validation_stats


if __name__ == "__main__":
    # Example usage
    print("Time aggregation module loaded successfully")
    print(f"tsam available: {TSAM_AVAILABLE}")
