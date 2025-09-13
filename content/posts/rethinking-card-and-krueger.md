---
author: "Sercan Ahi"
title: "From Difference-in-Differences to Synthetic Control: Rethinking Card & Krueger (1994)"
date: "2025-09-13"
---

First, a big thank you to [Aaron Mamula's thoughtful replication](https://aaronmams.github.io/Card-Krueger-Replication/) of Card & Krueger's classic 1994 minimum-wage study. His R-based walk-through of the survey data, regression results, and sensitivity checks are clear and pedagogically valuable. Whether you are a newcomer to the causal inference research literature, or a data analytics professional interested in transparent, reproducible workflows, Aaron's replication is one of the best references. It shows how careful design and open code can bring an iconic study back to life.

Revisiting classical studies like this always raises the same question for me:

> _"What would a modern toolkit make of it?"_

## Why Card & Krueger (1994) Mattered

When David Card and Alan Krueger compared fast-food employment in New Jersey (which raised its minimum wage) against Pennsylvania (which did not), their finding was revolutionary: **employment did not fall**. In fact, it rose slightly.

The study challenged decades of textbook predictions, where a binding minimum wage was assumed to cut jobs. More importantly, it helped launch the [credibility revolution](https://en.wikipedia.org/wiki/Credibility_revolution) in economics, a move toward natural experiments and data-driven causal inference.

## The Donor Pool Dilemma

Card & Krueger's design was essentially a **two-state difference-in-differences**. Simple, elegant, but fragile. What if Pennsylvania was not a perfect counterfactual for New Jersey? What if spillovers in border labor markets contaminated the comparison?

This is where [Synthetic Control Method](https://en.wikipedia.org/wiki/Synthetic_control_method) (SCM) can come in. Instead of betting everything on a single comparator, SCM builds a weighted average of multiple donor states, chosen so that their **pre-treatment employment paths** closely resemble New Jersey's. Post-treatment, the divergence between NJ and its **synthetic twin** is interpreted as the policy effect.

## Modern Tools for a Classic Case

Today, using [Synth](https://cran.r-project.org/web/packages/Synth/index.html) in R or [cvxpy](https://github.com/cvxpy/cvxpy) in Python, we can:

- **Construct synthetic New Jersey**: combining donors like NY, CT, MA, and DE with weights chosen by optimization.
- **Match richer predictors**: not just pre-period employment, but industry composition, wage levels, demographic traits.
- **Test robustness**: by running placebo SCMs (pretend each donor was treated) and comparing effect sizes.
- **Improve inference**: using bootstrap or wild-cluster standard errors rather than simple OLS SEs.

In other words, SCM does not just re-estimate Card & Krueger, it can even reframe the entire exercise with a different lens on counterfactual construction.

## A Timeline of the Debate

The Card & Krueger study triggered decades of replications, criticisms, and methodological advances:

- 1994: Original NJ vs PA study (_Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania_) finds no job loss.
- 2000: Neumark & Wascher study (Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania: Comment) finds job losses using payroll data.
- 2000: Card & Krueger response (Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania: Reply) reanalyze same payroll data and conclude that results hinge on specification.
- 2010s: Meta-analyses & new causal inference tools hint that most effects found to be small.
- 2021: David Card wins Nobel Prize, and this study becomes emblematic of economics' empirical turn.

## Show Me Some Code

Refer to the `main()` function in the script below to get a sense of what is happening.

```python
"""Application of Synthetic Control Method on Card & Krueger (1994)."""

from typing import Iterable, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd


def create_fake_panel(
    treated: str = "NJ",
    donors: Sequence[str] = ("NY", "CT", "MA", "DE", "MD"),
    years: Iterable[int] = range(1989, 1995),
    treat_start: int = 1992,
    effect_size: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a fake panel dataset reminiscent of Card & Krueger (1994).

    - The dataset has one treated unit (e.g., 'NJ') and
        several donor units, such as 'NY' or 'CT'
    - The outcome is 'employment'
    - We also fabricate two time-invariant covariates per unit:
        'avg_wage_pre' and 'industry_share'.
        avg_wage_pre: average wage in the target industry (before treatment)
        industry_share: share of establishments in a target industry

    Args
    ----
    treated (str): Name of the treated unit
    donors (Sequence[str]): Names of donor units (untreated)
    years (Iterable[int]): Sequence of integer years (e.g., 1989..1994)
    treat_start (int): First year in which the treatment takes effect
        for the treated unit
    effect_size (float): Additive treatment effect applied to the treated unit
        in post-treatment years
    seed (int): Random seed for reproducibility

    Returns
    -------
    pd.DataFrame: Long-form panel dataset
    """
    rng = np.random.default_rng(seed)
    years = list(int(y) for y in years)
    year_idx = np.array(years) - min(years)

    def gen_series(base: float, slope: float) -> np.ndarray:
        eps = rng.normal(0.0, 0.8, size=len(years))
        smooth = np.linspace(0.0, 1.2, len(years))
        return base + slope * year_idx + smooth + eps

    rows: list[dict] = []
    for u in (treated, *donors):
        base = rng.normal(30.0, 2.5)
        slope = rng.normal(0.20, 0.05)
        y = gen_series(base, slope)

        # Apply a positive treatment effect to the treated unit
        # from treat_start onward
        if u == treated:
            for i, yr in enumerate(years):
                if yr >= treat_start:
                    y[i] += effect_size

        # Time-invariant covariates for this toy example
        avg_wage_pre = rng.normal(8.2, 0.4)
        industry_share = rng.uniform(0.25, 0.55)

        for n_yr, yr in enumerate(years):
            rows.append(
                {
                    "unit": u,
                    "year": int(yr),
                    "employment": float(y[n_yr]),
                    "avg_wage_pre": float(avg_wage_pre),
                    "industry_share": float(industry_share),
                }
            )

    return pd.DataFrame(rows)


def assemble_design(
    df: pd.DataFrame,
    treated: str,
    donors: Sequence[str],
    pre_years: Sequence[int],
    covariate_cols: Sequence[str] = ("avg_wage_pre", "industry_share"),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare the input tables that SCM needs.

    Args
    ----
    df (pd.DataFrame): Long-form panel dataset
    treated (str): Name of the treated unit in `df['unit']`
    donors (Sequence[str]): Names of donor units
    pre_years (Sequence[int]): Years used to build pre-treatment predictors
    covariate_cols (Sequence[str]): Additional predictors

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        - design: table of predictors for each unit (used to fit weights)
        - Y_wide: table of employment by unit-year (used to build trajectories)
        - predictor_cols: list of predictor column names
    """
    # Wide outcomes: one column per year
    Y_wide = df.pivot(index="unit", columns="year", values="employment")
    Y_wide.columns = [f"Y_{c}" for c in Y_wide.columns]

    # Covariates: take first observed per unit
    covars = (
        df[["unit", *covariate_cols]]
        .drop_duplicates("unit")
        .set_index("unit")
    )

    # Join into a single design matrix
    design = Y_wide.join(covars, how="left")

    # Pre-treatment trajectory predictors
    pre_cols = [f"Y_{t}" for t in pre_years]
    predictor_cols = [*pre_cols, *covariate_cols]

    # Basic checks
    if treated not in design.index:
        raise ValueError(f"Treated unit '{treated}' not found in data.")
    for u in donors:
        if u not in design.index:
            raise ValueError(f"Donor unit '{u}' not found in data.")

    return design, Y_wide, predictor_cols


def fit_scm_weights(
    design: pd.DataFrame,
    treated: str,
    donors: Sequence[str],
    predictor_cols: Sequence[str],
    alpha: Sequence[float] | None = None,
    ridge: float = 1e-4,
    solver: str = "OSQP",
) -> pd.Series:
    """Solve SCM weight optimization with an L2 (least-squares) objective.

    Objective:
        minimize ||A (X0 w - X1)||_2^2 + ridge * ||w||_2^2
        s.t.     sum(w) = 1, w >= 0

    where:
        - X1 is the treated predictor vector,
        - X0 is the donor predictor matrix,
        - A is a diagonal matrix of predictor importances (alpha).

    Args
    ----
    design (pd.DataFrame): Unit-indexed DataFrame containing predictor_cols
    treated (str): Treated unit name
    donors (Sequence[str]): Donor unit names (columns/rows must exist in design)
    predictor_cols (Sequence[str]): Names of columns in `design`
        used as predictors.
    alpha (Sequence[float] | None): Optional predictor importances.
        Length must equal len(predictor_cols).
        If None, all predictors are weighted equally (1.0).
    ridge (float): Ridge penalty to stabilize weights (set to 0.0 to disable)
    solver (str): CVXPy solver name (e.g., "OSQP", "ECOS", "SCS").

    Returns
    -------
    pd.Series: Donor weights indexed by donor unit name (non-negative, sum to 1)
    """
    X1 = design.loc[treated, predictor_cols].to_numpy(float)  # (P,)
    X0 = design.loc[donors, predictor_cols].to_numpy(float).T  # (P x J)

    P, J = X0.shape
    w = cp.Variable(J, nonneg=True)

    if alpha is None:
        A = np.eye(P)
    else:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape[0] != P:
            raise ValueError("alpha length must match number of predictors.")
        A = np.diag(alpha)

    objective = cp.Minimize(
        cp.sum_squares(A @ (X0 @ w - X1)) + ridge * cp.sum_squares(w)
    )
    constraints = [cp.sum(w) == 1.0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    w_val = np.clip(w.value, 0.0, None)
    s = float(w_val.sum()) if w_val is not None else 0.0
    if s <= 0.0:
        # Fallback to uniform weights if solver fails or degenerates
        w_val = np.ones(J) / J
    else:
        w_val = w_val / s

    return pd.Series(w_val, index=list(donors))


def build_synthetic_series(
    Y_wide: pd.DataFrame,
    treated: str,
    donors: Sequence[str],
    weights: pd.Series,
    all_years: Sequence[int],
) -> tuple[pd.Series, pd.Series]:
    """Construct treated and synthetic outcome time series for all years.

    Args
    ----
    Y_wide (pd.DataFrame): Unit x year wide matrix of the outcome 'employment'
        (columns named 'Y_<year>')
    treated (str): Treated unit name.
    donors (Sequence[str]): Donor unit names
    weights (pd.Series): Donor weights
        (index must match donor names, values sum to 1)
    all_years (Sequence[int]): All years to include in the resulting series

    Returns
    -------
    tuple[pd.Series, pd.Series]: (treated_series, synthetic_series),
        both indexed by integer years
    """
    year_cols = [f"Y_{y}" for y in all_years]
    treated_series = pd.Series(
        Y_wide.loc[treated, year_cols].to_numpy(float),
        index=list(all_years),
        name="treated",
    )

    donor_mat = Y_wide.loc[donors, year_cols].to_numpy(float)  # (J x T)
    synth = donor_mat.T @ weights.loc[list(donors)].to_numpy()  # (T,)
    synthetic_series = pd.Series(synth, index=list(all_years), name="synthetic")

    return treated_series, synthetic_series


def placebo_gaps(
    design: pd.DataFrame,
    Y_wide: pd.DataFrame,
    treated: str,
    donors: Sequence[str],
    predictor_cols: Sequence[str],
    post_years: Sequence[int],
    alpha: Sequence[float] | None = None,
    ridge: float = 1e-4,
    solver: str = "OSQP",
) -> pd.DataFrame:
    """Compute simple placebo post-treatment gaps by rotating the treated unit.

    For each donor unit d:
      - treat d as if it were the treated unit,
      - fit SCM weights from the remaining units,
      - compute (d_actual - d_synthetic) in post-treatment years.

    Args
    ----
    design (pd.DataFrame): Unit-indexed DataFrame containing predictor_cols
    Y_wide (pd.DataFrame): Unit x year wide matrix of the outcome 'employment'
        (columns 'Y_<year>')
    treated (str): Name of the original treated unit
    donors (Sequence[str]): Donor unit names
    predictor_cols (Sequence[str]): Names of columns in `design`
        used as predictors
    post_years (Sequence[int]): Post-treatment years used to compute gaps
    alpha (Sequence[float] | None): Optional predictor importances;
        if None, all predictors weighted equally
    ridge (float): Ridge penalty for SCM fit
    solver (str): CVXPy solver name

    Returns
    -------
    pd.DataFrame: placebo gaps with index=post_years and columns=placebo units
    """
    gaps = {}
    for placebo in donors:
        pool = [u for u in (treated, *donors) if u != placebo]
        donor_pool = [u for u in pool if u != placebo]
        w_pl = fit_scm_weights(
            design=design,
            treated=placebo,
            donors=donor_pool,
            predictor_cols=predictor_cols,
            alpha=alpha,
            ridge=ridge,
            solver=solver,
        )

        year_cols = [f"Y_{y}" for y in post_years]
        y_trt = Y_wide.loc[placebo, year_cols].to_numpy(float)  # (T_post,)
        y_syn = (
            Y_wide
            .loc[w_pl.index, year_cols]
            .to_numpy(float).T @ w_pl.to_numpy()
        )
        gaps[placebo] = y_trt - y_syn

    return pd.DataFrame(gaps, index=list(post_years))


def main() -> None:
    """Orchestrate the process.

    1. Create a simple, believable dataset with one treated state (NJ) and
        several donor states (NY, CT, MA, DE, MD).
    2. Define what good matching means before the policy change (pre-treatment),
        using the prior years' outcomes and a couple of covariates.
    3. Ask an optimizer to build a "synthetic NJ" by combining donors
        with weights that best reproduce NJ's pre-treatment behavior.
    4. Compare NJ to its synthetic twin after the policy and
        read the difference as the policy effect.
    5. As a rough credibility check, we run "placebo" versions
        of the same exercise, pretending each donor were treated, to see
        if NJ's pattern looks special.
    """
    # 1) Data: a small, realistic toy example
    treated = "NJ"
    donors = ("NY", "CT", "MA", "DE", "MD")
    years = list(range(1989, 1995))  # 1989–1994
    pre_years = (1989, 1990, 1991)
    post_years = (1992, 1993, 1994)

    df = create_fake_panel(
        treated=treated,
        donors=donors,
        years=years,
        treat_start=1992,
        effect_size=1.0,
        seed=123,
    )

    # 2) Predictors: match the past to predict a better counterfactual
    design, Y_wide, predictor_cols = assemble_design(
        df=df, treated=treated, donors=donors, pre_years=pre_years
    )

    # Emphasize matching the pre-period outcomes a bit more than covariates
    alpha = np.ones(len(predictor_cols))
    # Downweight the last two predictors (our toy covariates)
    alpha[-2:] = 0.5

    # 3) Fit: let the optimizer choose donor weights
    # that **best match pre-treatment NJ**
    weights: pd.Series = fit_scm_weights(
        design=design,
        treated=treated,
        donors=donors,
        predictor_cols=predictor_cols,
        alpha=alpha,
        ridge=1e-4,
        solver="OSQP",
    ).sort_values(ascending=False)

    print("Donor weights (how much each state contributes to 'synthetic NJ'):")
    print(weights.round(3), end="\n\n")

    # 4) Effect: compare actual NJ to synthetic NJ after the policy
    treated_series, synthetic_series = build_synthetic_series(
        Y_wide=Y_wide,
        treated=treated,
        donors=donors,
        weights=weights,
        all_years=years,
    )

    gaps_post = (treated_series - synthetic_series).loc[list(post_years)]
    print("Post-treatment gaps (Treated − Synthetic):")
    print(gaps_post.round(3), end="\n\n")

    # 5) Placebos: does NJ look special relative to donors treated "as if"?
    # We can compare gaps_post with placebo_df visually or a pseudo p-value,
    # which is not implemented in this script.
    placebo_df = placebo_gaps(
        design=design,
        Y_wide=Y_wide,
        treated=treated,
        donors=donors,
        predictor_cols=predictor_cols,
        post_years=post_years,
        alpha=alpha,
        ridge=1e-4,
        solver="OSQP",
    )

    print("Placebo gaps (rows=years, columns=fake-treated donors):")
    print(placebo_df.round(3))


if __name__ == "__main__":
    main()
```

## Where This Leaves Us

What I find most inspiring about revisiting Card & Krueger's study, whether through Aaron's replication or through synthetic control extensions, is how it teaches us two timeless lessons:

1. **Empirical claims live or die by counterfactuals.** Choosing "what would have happened otherwise" is the central challenge, whether you pick Pennsylvania in 1994 or a convex combination of donors in 2025.
2. **Methods evolve, debates persist.** Each round of critique and re-analysis strengthens the field and pushes us toward better data, more robust tools, and a more nuanced understanding of the problem at hand.

## Closing Thought

Card & Krueger (1994) was not just about minimum wages. It was a turning point in how economics asks questions, how it argues with evidence, and how it learns from replication. Aaron Mamula's R-based blog post reminds us that old debates can spark new insights, especially when we apply today's methods to yesterday's puzzles.
