"""
Author: Wai Ip
Create Date: 2026-03-22
Last Update: 2026-03-22
Description: Descriptive and numerical statistical functions.
"""


import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import average_precision_score


def contingency_table(df: pd.DataFrame,
                      idx_group: list,
                      col_group: list,
                      is_pct: bool = False,
                      top_n: int = None,
                      ) -> pd.DataFrame:
    """
    Create contingency table with (COLUMN, CATEGORY) multi-index columns and index from DataFrame between two groups of
    categorical features.

    Args:
        df: Input Pandas DataFrame.
        idx_group: Index categories in output table selected from a list of columns in `df`.
        col_group: Column categories in output table selected from a list of columns in `df`.
        is_pct: If `True`, return the percentage of observations in fraction.
        top_n: Keep only the most frequent `top_n` labels in each feature.

    Returns:
        Multi-index contingency table in Pandas DataFrame.

    Example:
        ```python
        >>> import seaborn as sns
        >>> df = sns.load_dataset('titanic')
        >>> contingency_table(df, ['survived'], ['deck', 'class'], True)

            deck                      A         B         C  ...         F                   G
            class                 First     First     First  ...    Second     Third     Third
                     survived                                ...
            survived 0         0.039409  0.059113  0.118227  ...  0.004926  0.019704  0.009852
                     1         0.034483  0.172414  0.172414  ...  0.034483  0.004926  0.009852
        ```
    """
    d = pd.crosstab([df[col] for col in idx_group], [df[col] for col in col_group])
    d = (d / d.sum().sum()) if is_pct else d

    if top_n is not None:
        idx_c = d.sum(axis=0).sort_values(ascending=False)[:(min(top_n, d.shape[1]))].index
        idx_i = d.sum(axis=1).sort_values(ascending=False)[:(min(top_n, d.shape[0]))].index
        d = d.loc[idx_i][idx_c]

    if len(idx_group) == 1:
        d.index = pd.MultiIndex.from_product([[d.index.name], d.index])
    if len(col_group) == 1:
        d.columns = pd.MultiIndex.from_product([[d.columns.name], d.columns])
    return d


def contingency_table_combined(df: pd.DataFrame,
                               is_pct: bool = False,
                               top_n: int = None,
                               ) -> pd.DataFrame:
    """
    Create contingency table with (COLUMN, CATEGORY) multi-index columns and index from DataFrame across between
    all individual columns combined in a single DataFrame.

    Args:
        df: Input Pandas DataFrame.
        is_pct: If `True`, return the percentage of observations in fraction.
        top_n: Keep only the most frequent `top_n` labels in each feature.

    Returns:
        Multi-index contingency table combined in Pandas DataFrame.

    Example:
        ```python
        >>> import seaborn as sns
        >>> df = sns.load_dataset('titanic')
        >>> contingency_table_combined(df[['deck', 'class', 'survived']])

                                    deck                        class              survived
                               A   B   C   D   E   F  G First Second Third        0    1
            deck     A        15   0   0   0   0   0  0    15      0     0        8    7
                     B         0  47   0   0   0   0  0    47      0     0       12   35
                     C         0   0  59   0   0   0  0    59      0     0       24   35
                     D         0   0   0  33   0   0  0    29      4     0        8   25
                     E         0   0   0   0  32   0  0    25      4     3        8   24
                     F         0   0   0   0   0  13  0     0      8     5        5    8
                     G         0   0   0   0   0   0  4     0      0     4        2    2
            class    First    15  47  59  29  25   0  0   216      0     0       80  136
                     Second    0   0   0   4   4   8  0     0    184     0       97   87
                     Third     0   0   0   0   3   5  4     0      0   491      372  119
            survived 0         8  12  24   8   8   5  2    80     97   372      549    0...
            ```
    """
    l = df.shape[1]
    holder = np.zeros((l, l)).tolist()
    layers = list()
    for r in range(l):
        for c in range(l):
            if c >= r:
                holder[r][c] = contingency_table(df, [df.columns[r]], [df.columns[c]], is_pct=is_pct, top_n=top_n)
            if c == (l - 1):
                if r == 0:
                    layers.append(pd.concat(holder[r], axis=1))
                else:
                    layers.append(pd.concat([pd.concat([holder[i][r] for i in range(r)], axis=0).T] + [holder[r][i] for i in range(r, l)], axis=1))

    return pd.concat(layers, axis=0)


def cramers_v(conf_matrix: pd.DataFrame,
              chi2: float = None,
              ) -> float:
    """
    Calculate Cramér's V statistics which measures the strength of association between two categorical variables.

    Args:
        conf_matrix: A contingency table with counts between two categorical features.
        chi2: Chi-square statistics. If `None`, calculate the value internally.

    Returns:
        Cramér's V value.
    """
    chi2 = chi2_contingency(conf_matrix)[0] if chi2 is None else chi2

    n = conf_matrix.sum().sum()
    k = min(conf_matrix.shape) - 1
    return np.sqrt(chi2 / (n * k))


def association(df: pd.DataFrame,
                cont_tbl: pd.DataFrame = None,
                ) -> pd.DataFrame:
    """
    Calculate association statistics including Chi-square statistics, p-value, degree of freedom, and
    Cramér's V statistics between all columns in a Pandas DataFrame. Summarize the values in table.

    Args:
        df: Input DataFrame with categorical features.
        cont_tbl: Contingency table with all feqtures in `df`. If `None`, create one from `contingency_table_combined()`
            internally.

    Returns:
        Table of association statistics with columns Chi-square statistics, p-value, degree of freedom,
        and Cramér's V statistics.
    """
    def helper(tbl: pd.DataFrame) -> tuple:
        cols = tbl.columns[~tbl.isna().all(axis=0)]
        if len(cols) > 1:
            tbl = tbl.loc[~tbl.isna().all(axis=1)][cols].fillna(0)
        else:
            tbl = tbl[cols[0]].to_frame(cols[0]).fillna(0)
        if not tbl.empty:
            chi2, p, dof, _ = chi2_contingency(tbl)
            cv = cramers_v(tbl, chi2)
            return chi2, p, dof, cv
        else:
            return None, None, None, None

    cont_tbl = contingency_table_combined(df) if cont_tbl is None else cont_tbl

    ls_asso = []
    for col_1 in df.columns:
        for col_2 in df.columns:
            if col_2 != col_1:
                ctbl = cont_tbl.loc[col_1][col_2]
                chi2, p, dof, cv = helper(ctbl)
                ls_asso.append([col_1, col_2, chi2, p, dof, cv])

    asso = pd.DataFrame(ls_asso, columns=['column_1', 'column_2', 'chi_sq', 'p_value', 'dof', 'cramers_v'])
    asso.sort_values(['column_1', 'column_2'], inplace=True)

    return asso.reset_index(drop=True)


def anova(df: pd.DataFrame,
          dep: Union[str, list],
          indep: Union[str, list],
          xdep: List[Tuple[str, str]] = None,
          cat_var: Union[str, list] = None,
          scale: float = None,
          test: str = 'F',
          typ: int = 2,
          robust: str = None
          ) -> pd.DataFrame:
    """
    Performs ANOVA (Analysis of Variance) on specified variables and creates a summary table.

    Args:
        df: Input Pandas DataFrame.
        dep: Dependent variable column.
        indep: Independent variable column(s).
        xdep: List of tuples containing cross dependency terms. For example,
            `[(a, b), (c, d)]` means assess whether the effect of `a` on the dependent variable depends on `b`,
            and similarly for `c` and `d`.
        cat_var: Categorical variables in `indep`. Default to `None`.
        scale : float
            Estimate of variance, If None, will be estimated from the largest
            model. Default is None.
        test : str {"F", "Chisq", "Cp"} or None
            Test statistics to provide. Default is "F".
        typ : str or int {"I","II","III"} or {1,2,3}
            The type of Anova test to perform. See notes.
        robust : {None, "hc0", "hc1", "hc2", "hc3"}
            Use heteroscedasticity-corrected coefficient covariance matrix.
            If robust covariance is desired, it is recommended to use `hc3`.

    Returns:
        ANOVA table with test statistics, p-values, and effect sizes.
    """
    # Adjustment for categorical names
    if cat_var is not None:
        cat_var = [cat_var] if isinstance(cat_var, str) else cat_var
        cat_var = [v for v in cat_var if v in indep or v == dep]

    # Adjustment for independent variable names
    indep = [indep] if isinstance(indep, str) else indep
    indep = [f'C({v})' if v in cat_var else v for v in indep] if cat_var is not None else indep
    indep = ' + '.join(indep)

    # Adjustment for dependent variable name
    dep = [dep] if isinstance(dep, str) else dep
    dep = [f'C({v})' if v in cat_var else v for v in dep] if cat_var is not None else dep
    assert len(dep) == 1, 'Does not accept multiple dependent variables. Use MANOVA instead.'
    dep = dep[0]

    # Adjustment for interaction pairs
    if xdep is not None:
        xdep = [[f'C({v})' if v in cat_var else v for v in pair] for pair in xdep] if cat_var is not None else xdep
        xdep = ' + '.join([':'.join(pair) for pair in xdep])
        indep = indep + ' + ' + xdep

    # Create linear model
    model = ols(f"{dep} ~ {indep}", data=df).fit()

    # Output ANOVA table
    result = anova_lm(model,
                      scale=scale,
                      test=test,
                      typ=typ,
                      robust=robust).reset_index()
    result.rename(columns={'index': 'attr'}, inplace=True)
    result['dep_var'] = dep
    return result[['dep_var', 'attr'] + result.columns.difference(['dep_var', 'attr']).to_list()]


def vif(df: pd.DataFrame,
        num_cols: Union[str, list] = None,
        ) -> pd.DataFrame:
    """
    Calculate the variance inflation factor of numerical features in a Pandas DataFrame.
    VIF measures how much variance of a regression coefficient is inflated due to multicollinearity.

    Note:
        - VIF > 10 → High multicollinearity (serious issue).
        - VIF > 5 → Moderate multicollinearity (potential issue).
        - VIF ≈ 1 → No multicollinearity.

    Args:
        df: Input Pandas DataFrame containing numerical values only without `Inf` or `NaN` values.
        num_cols: Numerical columns in `df`.

    Returns:
        VIF table in Pandas DataFrame.
    """
    num_cols = [num_cols] if isinstance(num_cols, str) else num_cols
    num_cols = df.columns.to_list() if num_cols is None else num_cols
    df_num = df[num_cols]
    result = [(df_num.columns[i], variance_inflation_factor(df_num, i)) for i in range(df_num.shape[1])]
    return pd.DataFrame(result, columns=['column', 'vif'])


def tukey_hsd(df: pd.DataFrame,
              dep: Union[str, list],
              indep: Union[str, list],
              alpha: float = 0.05,
              ) -> pd.DataFrame:
    """
    Performs Tukey's HSD (Honestly Significant Difference) test on specified variables.
    Usually used as post-hoc analysis after ANOVA analysis.

    Args:
       df: Input DataFrame containing the variables to analyze with `NaN` handled.
       dep: Dependent variable(s) - column name(s) containing numerical data.
       indep: Independent variable(s) - column name(s) containing categorical data.
       alpha: Significance level for the test. Defaults to `0.05`.

    Returns:
       Results of Tukey's HSD test in Pandas DataFrame containing:

           - dep_var: Name of dependent variable
           - ind_var: Name of independent variable
           - group1, group2: Pairs of groups being compared
           - meandiff: Mean difference between groups
           - p-adj: Adjusted p-value
           - lower, upper: Confidence interval bounds
           - reject: Boolean indicating if null hypothesis was rejected

    Example:
        ```python
        >>> import seaborn as sns
        >>> df = sns.load_dataset('titanic')
        >>> df['age'].fillna(df['age'].mean(), inplace=True)
        >>> df[['class', 'sex']].fillna(df[['class', 'sex']].mode(), inplace=True)
        >>> tukey_hsd(df, 'age', ['class', 'sex'])

              dep_var ind_var  group1  group2    lower  meandiff   p-adj  reject   upper
            0     age   class   First  Second -10.0684   -7.1812  0.0000    True -4.2939
            1     age   class   First   Third -12.9947  -10.6449  0.0000    True -8.2951
            2     age   class  Second   Third  -5.9514   -3.4637  0.0032    True -0.9760
            3     age     sex  female    male   0.5049    2.2891  0.0120    True  4.0733
        ```
    """
    dep = [dep] if isinstance(dep, str) else dep
    indep = [indep] if isinstance(indep, str) else indep
    result = []
    for d in dep:
        for ind in indep:
            tukey = pairwise_tukeyhsd(df[d], df[ind], alpha=alpha)
            tukey = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey['dep_var'] = d
            tukey['ind_var'] = ind
            result.append(tukey[['dep_var', 'ind_var'] + tukey.columns.difference(['dep_var', 'ind_var']).to_list()])
    return pd.concat(result, axis=0, ignore_index=True)


def paired_boostrap_prauc_diff(y_true,
                               y_prob_a,
                               y_prob_b,
                               n_bootstrap=5000,
                               confidence_level=0.95,
                               random_state=42
                               ):
    """
    Calculate paired bootstrap performance differences between models in terms of PRAUC.

    Args:
        y_true: true outcome
        y_prob_a: prediction probabilities from model a
        y_prob_b: prediction probabilities from model b
        n_bootstrap: number of bootstrap samples
        confidence_level: confidence level
        random_state: random seed

    Returns:
        Bootstrapped results
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    score_a = average_precision_score(y_true, y_prob_a)
    score_b = average_precision_score(y_true, y_prob_b)
    observed_diff = score_b - score_a

    bootstrap_diff: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

        y_boot = y_true[indices]

        if np.unique(y_boot).size < 2:
            continue

        score_a_boot = average_precision_score(
            y_boot,
            y_prob_a[indices],
        )
        score_b_boot = average_precision_score(
            y_boot,
            y_prob_b[indices],
        )

        bootstrap_diff.append(score_b_boot - score_a_boot)

    bs_diff = np.asarray(bootstrap_diff)

    # if differences.size == 0:
    #     raise RuntimeError("No valid bootstrap samples were generated.")

    alpha = 1.0 - confidence_level
    ci_lower, ci_upper = np.quantile(bs_diff,[alpha / 2, 1 - alpha / 2])

    return dict(
        score_a=float(score_a),
        score_b=float(score_b),
        observed_diff=float(observed_diff),
        bootstrap_diff=bs_diff,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        prob_b_better=float(np.mean(bs_diff > 0)),
    )

def compare_adjacent_models(y_true,
                            dict_prob,
                            n_bootstrap=5000,
                            confidence_level=0.95,
                            random_state=42
                            ):
    """
    Compare PRAUC across models in order in predictions probability dictionary.

    Args:
        y_true: true outcome
        dict_prob: dictionary of model probabilities
        n_bootstrap: number of bootstrap samples
        confidence_level: confidence level
        random_state: random seed

    Returns:
        Bootstrapped results
    """
    names = list(dict_prob.keys())
    rows = []

    for i in range(len(names) - 1):
        model_a = names[i]
        model_b = names[i + 1]

        result = paired_boostrap_prauc_diff(
            y_true=y_true,
            y_prob_a=dict_prob[model_a],
            y_prob_b=dict_prob[model_b],
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state + i,
        )

        rows.append({
            'comparison': f'{model_b} - {model_a}',
            'model_a_prauc': result['score_a'],
            'model_b_prauc': result['score_b'],
            'observed_diff': result['observed_diff'],
            'bootstrap_diff': result['bootstrap_diff'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'ci_ex_zero': ((result['ci_lower'] > 0) or (result['ci_upper'] < 0)),
            'prob_b_better': result['prob_b_better'],
        })

    return pd.DataFrame(rows)
