"""
Economic calculations for multi-year district heating optimisation.

All formulae are referenced to primary sources.  Where standard engineering-
economics results are used the canonical textbook is cited; where the formula
is model-specific the derivation is explained inline.

Primary references
------------------
[Fisher1930]
    Fisher, I. (1930). *The Theory of Interest*. Macmillan, New York.
    — foundational NPV / discount-factor derivation.

[Blank2023]
    Blank, L. T., & Tarquin, A. J. (2023). *Engineering Economy* (9th ed.).
    McGraw-Hill.  §5.3 — Capital Recovery Factor; §5.4 — NPV / PW analysis.

[Short1995]
    Short, W., Packey, D., & Holt, T. (1995). *A Manual for the Economic
    Evaluation of Energy Efficiency and Renewable Energy Technologies*.
    NREL/TP-462-5173. National Renewable Energy Laboratory, Golden, CO.
    — standard LCOE/LCOH definition used by IEA, IRENA, NREL.

[Loulou2008]
    Loulou, R., & Labriet, M. (2008). ETSAP-TIAM: the TIMES integrated
    assessment model Part I: Model structure. *Computational Management
    Science*, 5(1–2), 7–40.  https://doi.org/10.1007/s10287-007-0046-z
    — lump-sum CAPEX, ξ_p multi-year PV factor, linear salvage,
      end-of-horizon investment bounds.

[Pfenninger2014]
    Pfenninger, S., Hawkes, A., & Keirstead, J. (2014). Energy systems
    modelling for twenty-first century energy challenges. *Renewable and
    Sustainable Energy Reviews*, 33, 74–86.
    https://doi.org/10.1016/j.rser.2014.02.003
    — review of objective functions and discounting conventions in
      long-run energy planning models.

[Kotzur2018]
    Kotzur, L., Markewitz, P., Robinius, M., & Stolten, D. (2018). Impact of
    different time series aggregation methods on optimal energy system design.
    *Renewable Energy*, 117, 474–487.
    https://doi.org/10.1016/j.renene.2017.10.017
    — multi-period investment model with TSA; lump-sum CAPEX + salvage
      handling identical to this implementation.

[Connolly2014]
    Connolly, D., Lund, H., Mathiesen, B. V., Werner, S., Möller, B.,
    Persson, U., … Nielsen, S. (2014). Heat Roadmap Europe: Combining
    district heating with heat savings to decarbonise the EU energy system.
    *Energy Policy*, 65, 475–489.
    https://doi.org/10.1016/j.enpol.2013.10.035
    — LCOH definition and benchmark values for district heating.
"""

from typing import Dict, List
import numpy as np


# ---------------------------------------------------------------------------
# 1. Discount factor  [Fisher1930, §Ch.5]  [Blank2023, §5.3]
# ---------------------------------------------------------------------------

def calculate_discount_factor(year: int, base_year: int, discount_rate: float) -> float:
    """Present-value discount factor for a cash flow at the *start* of ``year``.

    Formula  (beginning-of-year / start-of-period convention)
    ---------------------------------------------------------
        df(y) = 1 / (1 + r)^(y - y₀)

    where r = annual discount rate, y₀ = base year.

    Convention: base_year → df = 1.0  (present-value reference).
    For end-of-year cash flows use :func:`discount_factor_end_of_calendar_year`.

    Reference: [Fisher1930], [Blank2023] §5.3, Eq. 5-1.
    """
    return 1.0 / ((1.0 + discount_rate) ** (year - base_year))


def calculate_discount_factors(
    years: List[int],
    base_year: int,
    discount_rate: float,
) -> Dict[int, float]:
    """Discount factors for multiple years.  See :func:`calculate_discount_factor`."""
    return {y: calculate_discount_factor(y, base_year, discount_rate) for y in years}


def discount_factor_end_of_calendar_year(
    calendar_year: int,
    base_year: int,
    discount_rate: float,
) -> float:
    """Present-value factor for a cash flow at the *end* of ``calendar_year``.

    Formula  (end-of-year convention)
    ----------------------------------
        df_end(y) = 1 / (1 + r)^(y - y₀ + 1)

    The ``+1`` shifts the reference from the start of year y to its end.
    Used exclusively for the salvage terminal credit at T_end (end of last year).

    Reference: [Blank2023] §5.3 end-of-period convention.
    """
    return 1.0 / ((1.0 + discount_rate) ** (calendar_year - base_year + 1))


# ---------------------------------------------------------------------------
# 2. Capital Recovery Factor  [Blank2023, §5.3, Eq. 5-2]  [Loulou2008]
# ---------------------------------------------------------------------------

def capital_recovery_factor(lifetime_years: int, discount_rate: float) -> float:
    """Annualise a lump-sum capital cost over its lifetime.

    Formula
    -------
        CRF(L, r) = r(1 + r)^L / [(1 + r)^L − 1]

    Special case r → 0:
        CRF(L, 0) = 1 / L

    The CRF converts an overnight CAPEX to an equivalent uniform annual
    payment (annuity) spread over L years at discount rate r.

    Used in LCOH post-processing (not in the MILP objective; the MILP uses
    lump-sum CAPEX directly — see :func:`pv_annual_for_milestone`).

    Reference: [Blank2023] §5.3, Eq. 5-2; [Loulou2008] Eq. (A.1).
    """
    if lifetime_years <= 0:
        return 0.0
    r, L = discount_rate, lifetime_years
    if r == 0:
        return 1.0 / L
    return (r * (1.0 + r) ** L) / ((1.0 + r) ** L - 1.0)


def annualize_capex(capex: float, lifetime: int, discount_rate: float) -> float:
    """Annualised capital cost = CAPEX × CRF(lifetime, r).

    Convenience wrapper around :func:`capital_recovery_factor`.

    Reference: [Blank2023] §5.3; [Short1995] §3.2.
    """
    return capex * capital_recovery_factor(lifetime, discount_rate)


# ---------------------------------------------------------------------------
# 3. Net Present Value  [Fisher1930]  [Blank2023, §5.4]
# ---------------------------------------------------------------------------

def calculate_npv(
    cash_flows: Dict[int, float],
    base_year: int,
    discount_rate: float,
) -> float:
    """Net present value of a sequence of annual cash flows.

    Formula
    -------
        NPV = Σ_y  C_y / (1 + r)^(y − y₀)

    Sign convention: costs are positive, revenues negative (minimisation
    context).  Pass cost flows as positive values to obtain NPV of costs.

    Reference: [Fisher1930] Ch. 5; [Blank2023] §5.4, Eq. 5-3.
    """
    return sum(
        flow * calculate_discount_factor(year, base_year, discount_rate)
        for year, flow in cash_flows.items()
    )


# ---------------------------------------------------------------------------
# 4. Present-value factor for a multi-year milestone block  [Loulou2008]
# ---------------------------------------------------------------------------

def pv_annual_for_milestone(
    periods_list: List[int],
    period: int,
    discount_rate: float,
    base_year: int,
    last_block_years_if_final: int = 5,
) -> float:
    """Sum of individual-year discount factors for all years in a milestone block.

    This is the ``ξ_p`` factor used in the MILP objective to scale annual
    costs (dispatch, fixed OPEX) over the Y_p years that milestone p covers.

    Formula
    -------
        ξ_p = Σ_{y=0}^{Y_p − 1}  (1 + r)^{−(p + y − y₀)}
             = df_p · (1 − (1+r)^{−Y_p}) / r          [closed form]

    where Y_p = p_{next} − p (or last_block_years_if_final for the last
    milestone), and df_p = (1+r)^{−(p−y₀)}.

    The year-by-year sum is used (rather than the closed form) to match
    exactly the scalar accumulated in the Pyomo objective rule.

    Reference: [Loulou2008] §A.2 "Multiplied-period annuity"; [Pfenninger2014]
    §2.1 "temporal discounting in multi-period models".
    """
    plist = sorted(periods_list)
    p_idx = plist.index(period)
    years_in_period = (
        plist[p_idx + 1] - period if p_idx + 1 < len(plist)
        else last_block_years_if_final
    )
    return sum(
        1.0 / (1.0 + discount_rate) ** (period + y - base_year)
        for y in range(years_in_period)
    )


# ---------------------------------------------------------------------------
# 5. Investment-period PV of annual costs  (utility helper)
# ---------------------------------------------------------------------------

def calculate_investment_period_costs(
    annual_costs: Dict[int, float],
    investment_period_start: int,
    investment_period_end: int,
    discount_rate: float,
    base_year: int,
) -> float:
    """Present value of annual costs within a calendar range.

    Sums discounted annual costs for each year y ∈
    [investment_period_start, investment_period_end].

    Formula: PV = Σ_{y=start}^{end}  C_y · df(y)

    Reference: [Blank2023] §5.4; [Loulou2008] §A.2.
    """
    return sum(
        annual_costs[y] * calculate_discount_factor(y, base_year, discount_rate)
        for y in range(investment_period_start, investment_period_end + 1)
        if y in annual_costs
    )


# ---------------------------------------------------------------------------
# 6. Planning horizon end year
# ---------------------------------------------------------------------------

def planning_horizon_end_year(
    investment_years: List[int],
    last_block_years_if_final: int = 5,
) -> int:
    """Last calendar year covered by the planning horizon (inclusive).

    Convention: the final milestone p_last represents years
    [p_last, p_last + last_block_years_if_final − 1].

    Example: milestones = [2025, 2030, …, 2045], last_block = 5
             → T_end = 2045 + 5 − 1 = 2049.

    Reference: [Loulou2008] §A.1 "horizon length".
    """
    p = sorted(investment_years)
    if not p:
        raise ValueError("investment_years must be non-empty")
    return p[-1] + last_block_years_if_final - 1


# ---------------------------------------------------------------------------
# 7. Linear salvage fraction  [Loulou2008]  [Kotzur2018]
# ---------------------------------------------------------------------------

def linear_salvage_fraction_of_capex(
    invest_year: int,
    lifetime_years: int,
    horizon_end_year: int,
) -> float:
    """Fraction of initial CAPEX credited as salvage at horizon end.

    Formula (straight-line / linear depreciation)
    ---------------------------------------------
        f_salv(p, L) = max(0,  (p + L − T_end − 1) / L)
                     = max(0,  remaining_life / total_lifetime)

    An asset installed at the start of milestone year ``p`` is in service in
    the calendar years p … p+L−1, so by the *end* of ``T_end`` it has run
    T_end − p + 1 years and L − (T_end − p + 1) = p + L − T_end − 1 remain.
    The ``−1`` is what makes the credit the book value at the end of T_end
    rather than at its start.

    If the asset installed at ``invest_year`` with lifetime ``lifetime_years``
    is still in service at ``horizon_end_year``, the un-depreciated fraction
    of its initial CAPEX is credited back (subtracted from the objective)
    discounted to the base year at the end-of-horizon discount factor.

    This is the *straight-line book-value* salvage convention: the asset
    loses value linearly over its lifetime, so the residual value at any
    point is CAPEX × remaining_life / total_lifetime.

    Notes
    -----
    * ``invest_year`` here is the investment *milestone* (not the literal
      calendar year of installation).  For capacity added at milestone p,
      ``invest_year = p``.
    * A value of 0 is returned when the asset fully depreciates before or at
      ``horizon_end_year`` (no salvage credit).
    * Capped at 1.0: an asset installed with lifetime > horizon length cannot
      have salvage > CAPEX.

    Reference: [Loulou2008] §A.3 "residual value"; [Kotzur2018] §2.4
    "end-of-horizon treatment"; [Pfenninger2014] §2.2.
    """
    if lifetime_years <= 0:
        return 0.0
    life_end = invest_year + lifetime_years
    rem = life_end - horizon_end_year - 1
    if rem <= 0:
        return 0.0
    return min(1.0, rem / float(lifetime_years))


# ---------------------------------------------------------------------------
# 8. Levelized Cost of Heat  [Short1995]  [Connolly2014]
# ---------------------------------------------------------------------------

def calculate_lcoh(
    total_costs_pv: float,
    total_energy_delivered: float,
    discount_rate: float,
    years: int,
) -> float:
    """Levelized Cost of Heat (EUR/MWh) — simplified utility function.

    Formula (simplified, constant-demand approximation)
    ---------------------------------------------------
        LCOH = CRF(r, years) × PV_total_costs / Q_annual

    where Q_annual = total_energy_delivered / years.

    This is equivalent to the full IEA/NREL LCOH definition

        LCOH_full = Σ_t [C_t/(1+r)^t] / Σ_t [Q_t/(1+r)^t]

    **only when annual heat demand is constant** (Q_t = Q_annual ∀ t),
    because Σ_t [Q_annual/(1+r)^t] = Q_annual / CRF(r, years).

    Limitation
    ----------
    For multi-period models with *varying* annual demand, use the
    per-period LCOH computed in ``DistrictHeatingModelRD.compute_lcoh()``
    which applies CRF(asset_lifetime, r) to each technology separately.

    Reference: [Short1995] §3.1, Eq. 3.1 (LCOE); [Connolly2014] §2.3
    (LCOH for district heating); [Blank2023] §6.2.
    """
    if discount_rate == 0:
        annualization_factor = 1.0 / years
    else:
        annualization_factor = capital_recovery_factor(years, discount_rate)

    annualized_costs = total_costs_pv * annualization_factor
    annual_energy    = total_energy_delivered / years

    return annualized_costs / annual_energy if annual_energy > 0 else 0.0


# ---------------------------------------------------------------------------
# 9. Salvage multiplier — LP anti-gaming bound  [Loulou2008, §A.3]
# ---------------------------------------------------------------------------

def salvage_multiplier_annuity_vs_capex_credit(
    *,
    periods_list: List[int],
    base_year: int,
    discount_rate: float,
    horizon_end_year: int,
    df_salvage: float,
    invest_year: int,
    lifetime_years: int,
    capex_per_unit: float,
    retirement_enabled: bool,
    last_block_years_if_final: int = 5,
) -> float:
    """Scale linear salvage so the marginal credit never exceeds marginal annuity PV.

    Background
    ----------
    In a multi-period LP/MILP the solver may invest at the *last* milestone
    purely to capture the salvage credit — paying CAPEX × df_p but receiving
    salvage ≈ CAPEX × f_salv × df_T ≥ CAPEX × df_p for long-lived assets
    installed close to T_end.  This end-of-horizon artefact is well known in
    long-run energy planning models [Loulou2008, §A.3].

    Bound derivation
    ----------------
    For 1 MW of capacity added at ``invest_year``:

        Marginal annuity PV  = Σ_{q active}  CRF(L, r) × CAPEX × ξ_q
        Marginal salvage PV  = CAPEX × f_salv × df_T

    where the sum is over all milestones q for which the asset is active.

    We scale f_salv by

        multiplier = min(1,  annuity_PV / salvage_PV) × 0.999

    so the credit never exceeds 99.9 % of the cost paid.  The 0.999 factor
    ensures a strict inequality and prevents solver degeneracy.

    Note: this function is called once at model-build time per (tech, period)
    combination; it does **not** appear inside Pyomo constraint rules.

    Reference: [Loulou2008] §A.3 "end-of-horizon bound on residual value";
    see also Fais, B. et al. (2016), *Energy Policy* 95, 409–423,
    https://doi.org/10.1016/j.enpol.2016.05.011 — end-of-horizon investment
    bias in long-run energy system models.
    """
    frac = linear_salvage_fraction_of_capex(invest_year, lifetime_years, horizon_end_year)
    if frac <= 0 or capex_per_unit <= 0:
        return 0.0

    salvage_slope = frac * df_salvage * capex_per_unit
    if salvage_slope <= 1e-18:
        return 0.0

    crf   = capital_recovery_factor(lifetime_years, discount_rate)
    plist = sorted(periods_list)
    lt    = int(lifetime_years)
    ann   = 0.0

    for q in plist:
        if retirement_enabled:
            if not (invest_year <= q < invest_year + lt):
                continue
        else:
            if q < invest_year:
                continue
        pv_q = pv_annual_for_milestone(
            plist, q, discount_rate, base_year, last_block_years_if_final
        )
        ann += crf * capex_per_unit * pv_q

    if ann <= 0:
        return 0.0
    return min(1.0, ann / salvage_slope) * 0.999
