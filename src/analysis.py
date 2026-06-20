"""
Author: Wai Ip
Create Date: 2026-03-22
Last Update: 2026-06-02
Description: Exploratory data analysis interfaces.
"""


import re
import pandas as pd
import numpy as np
from typing import Union
from statsmodels.stats.descriptivestats import describe
from src.stats import contingency_table_combined, association


TYPE_NA = 'null'
TYPE_INF = 'inf'
TYPE_TEXTUAL = 'textual'
TYPE_CONTINUOUS = 'continuous'
TYPE_DISCRETE = 'discrete'
TYPE_CATEGORICAL = 'category'
TYPE_BINARY = 'binary'
TYPE_DATETIME = 'datetime'
TYPE_TIMESTAMP = 'timestamp'
TYPE_MIXED = 'mixed'


def _infer_dtype(d,
                 t,
                 th=0.2):

    def is_quantitative(d, t):
        return True if ('float' in t) or ('int' in t) or ('complex' in t) else False

    def is_discrete(d, t, tolerance=1e-9):
        # Check if all values are integers
        if 'int' in t:
            return True

        # Check if values are float but all are whole numbers (e.g., 1.0, 2.0)
        if ('float' in t) and (d.dropna() % 1 - 0 < tolerance).all():
            return True

        # Check if values have a fixed step size
        sorted_values = np.sort(d.dropna().unique())  # Remove NaN and sort
        diffs = np.diff(sorted_values)  # Compute differences

        if np.all(np.abs(diffs - np.median(diffs)) < tolerance):  # Check if differences are nearly constant
            return True

        return False

    def is_datetime(d, t):
        return True if ('datetime' in t) else False

    def is_timestamp(d, t):
        return True if ('timestamp' in t) else False

    def is_binary(d, t):
        return True if ('bool' in t) or (len(d.unique()) == 2) else False

    def is_categorical(d, t, th=0.2):
        return True if (len(d.unique()) <= round(len(d) * th)) and ('float' not in t) and ('complex' not in t) else False

    def is_null(t):
        return t == TYPE_NA

    if is_null(t):
        return TYPE_NA

    is_quant = is_quantitative(d, t)
    if is_quant:
        if is_binary(d, t):
            return TYPE_BINARY
        elif is_categorical(d, t, th):
            return TYPE_CATEGORICAL
        elif is_discrete(d, t):
            return TYPE_DISCRETE
        else:
            return TYPE_CONTINUOUS
    else:
        if is_datetime(d, t):
            return TYPE_DATETIME
        elif is_timestamp(d, t):
            return TYPE_TIMESTAMP
        elif is_binary(d, t):
            return TYPE_BINARY
        elif is_categorical(d, t, th):
            return TYPE_CATEGORICAL
        else:
            return TYPE_TEXTUAL


def dtype(df: pd.DataFrame,
          is_pct: bool = False,
          ) -> pd.DataFrame:
    """
    Get data type for each column in Pandas DataFrame. Categorize missing data by `Pandas.DataFrame.isna()` into `N/A`.

    Args:
        df: Input DataFrame with feature as columns.
        is_pct: If `True`, for each data type, return the percentage of observations in fraction instead of counts.

    Returns:
        Data type statistics in DataFrame, either count or proportion depends on `is_pct`.

    Example:
        ```python
        >>> import seaborn as sns
        >>> df = sns.load_dataset('iris')
        >>> dtype(df, is_pct=False)

                   sepal_length  sepal_width  petal_length  petal_width  species
            float         150.0        150.0         150.0        150.0      0.0
            str             0.0          0.0           0.0          0.0    150.0
        ```
    """
    dtypes = df.map(type)
    dtypes_str = dtypes.map(str)
    dtypes_str.values[df.isna()] = TYPE_NA
    dtypes_str.values[(df == np.inf) | (df == -np.inf)] = TYPE_INF
    dtypes_str.index.name = None

    ls_stat = list()
    for col, series in dtypes_str.items():
        tmp = dtypes_str[[col]].reset_index()
        ls_stat.append(tmp.groupby(col).count().rename(columns={tmp.columns[0]: col}))

    stat = pd.concat(ls_stat, axis=1).fillna(0)
    stat = stat / df.shape[0] if is_pct else stat
    stat.index = [re.search(r"<class '([\w]+)'>", s).group(1).lower() if re.search(r"<class '([\w]+)'>", s) else s for s in stat.index]
    stat.index = [re.search(r"<class '([\w.]+)\.(\w+)'>", s).group(2).lower() if re.search(r"<class '([\w.]+)\.(\w+)'>", s) else s for s in stat.index]
    return stat.sort_index()


def ftype(df: pd.DataFrame,
          dtypes: Union[pd.DataFrame, dict] = None,
          unique_th: float = 0.2,
          ) -> pd.DataFrame:
    """
    Get data feature type by inference based on data type and uniqueness.
    Each feature is assigned to only one feature type.

    Note:
        Currently the module contain a number of pre-defined feature types: 'textual', 'continuous', 'discrete',
        'category', 'binary', 'mixed', 'datetime', and 'timestamp'.

    Args:
        df: Input Pandas DataFrame with feature as columns.
        dtypes: Pandas DataFrame from `dtype()` or it's dictionary form. If `None`, create it internally.
        unique_th: Data uniqueness threshold. For example, `unique_th=0.2` means that if the number of unique value
            in a columns is less than or equaly to 20% of the number observation, the data is considered as categorical.

    Returns:
        Feature type DataFrame with column names.

    Example:
        ```python
        >>> import seaborn as sns
        >>> df = sns.load_dataset('iris')
        >>> ftype(df)

                              ftype
            column
            survived         binary
            pclass         category
            sex              binary
            age          continuous
            sibsp          category
            parch          category
            fare         continuous
            embarked       category
            class          category
            who            category
            adult_male       binary
            deck           category
            embark_town    category
            alive            binary
            alone            binary
        ```
    """
    if dtypes is None:
        dtypes = dtype(df)

    if isinstance(dtypes, pd.DataFrame):
        dtypes = dtypes.to_dict()

    ls_ftype = list()
    for col, dict_dtype in dtypes.items():

        ls_dtype = [k for k, v in dict_dtype.items() if (v != 0) and (k != TYPE_NA)]
        if len(ls_dtype) > 1:
            ls_dtype = [TYPE_MIXED]
        elif not ls_dtype:
            ls_dtype = [TYPE_NA]
        t = ls_dtype[0]
        d = df[col].dropna()
        ls_ftype.append((col, _infer_dtype(d, t, unique_th)))

    return pd.DataFrame(ls_ftype, columns=['column', 'ftype']).set_index('column')


class Univariate(object):
    """
    Univariate analysis. Calculate descriptive statistics of an input DataFrame.

    Args:
        df: Input Pandas DataFrame.
        ftypes: Feature type DataFrame from `ftype()`. If `None`, create it internally.
        dtypes: Data type Pandas DataFrame from `dtype()` or it's dictionary form. If `None`, create it internally.
        unique_th: Data uniqueness threshold. For example, `unique_th=0.2` means that if the number of unique value
            in a columns is less than or equaly to 20% of the number observation, the data is considered as categorical.
        para: Optional parameter in dictionary for `statsmodels.stats.descriptivestats.describe`.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 ftypes: pd.DataFrame = None,
                 dtypes: Union[pd.DataFrame, dict] = None,
                 unique_th: float = 0.2,
                 para: dict = None,
                 ) -> None:
        ftypes = ftype(df, dtypes=dtypes, unique_th=unique_th) if ftypes is None else ftypes
        para = dict() if para is None else para
        cat_cols = ftypes.loc[ftypes.values == TYPE_CATEGORICAL].index.to_list()
        bin_cols = ftypes.loc[ftypes.values == TYPE_BINARY].index.to_list()
        cat_cols += bin_cols
        for col in cat_cols:
            df[col] = df[col].astype('category')

        self.df = df
        self.ftypes = ftypes
        self.dtypes = dtypes
        self.unique_th = unique_th
        self.para = para

    def describe(self) -> pd.DataFrame:
        """
        Get the descriptive statistics of an input DataFrame.

        Returns:
            Descriptive statistics in Pandas DataFrame.

        Example:
            ```python
            >>> import seaborn as sns
            >>> df = sns.load_dataset('iris')
            >>> ftype = ftype(df)
            >>> u = Univariate(df, ftype)
            >>> u.describe()

                                  sepal_length  sepal_width  ...  petal_width     species
                nobs                150.000000   150.000000  ...   150.000000         150
                missing               0.000000     0.000000  ...     0.000000           0
                mean                  5.843333     3.057333  ...     1.199333         NaN
                std_err               0.067611     0.035588  ...     0.062236         NaN
                upper_ci              5.975849     3.127085  ...     1.321315         NaN
                lower_ci              5.710818     2.987581  ...     1.077352         NaN
                std                   0.828066     0.435866  ...     0.762238         NaN
                iqr                   1.300000     0.500000  ...     1.500000         NaN
                iqr_normal            0.963691     0.370651  ...     1.111952         NaN
                mad                   0.687556     0.336782  ...     0.658133         NaN
                mad_normal            0.861723     0.422094  ...     0.824848         NaN
                coef_var              0.141711     0.142564  ...     0.635551         NaN
                range                 3.600000     2.400000  ...     2.400000         NaN
                max                   7.900000     4.400000  ...     2.500000         NaN
                min                   4.300000     2.000000  ...     0.100000         NaN
                skew                  0.311753     0.315767  ...    -0.101934         NaN
                kurtosis              2.426432     3.180976  ...     1.663933         NaN
                jarque_bera           4.485875     2.697424  ...    11.416490         NaN
                jarque_bera_pval      0.106146     0.259574  ...     0.003318         NaN
                mode                  5.000000     3.000000  ...     0.200000         NaN
                mode_freq             0.066667     0.173333  ...     0.193333         NaN
                median                5.800000     3.000000  ...     1.300000         NaN
                distinct                   NaN          NaN  ...          NaN           3
                top_1                      NaN          NaN  ...          NaN      setosa
                top_2                      NaN          NaN  ...          NaN  versicolor
                top_3                      NaN          NaN  ...          NaN   virginica
                top_4                      NaN          NaN  ...          NaN        None
                top_5                      NaN          NaN  ...          NaN        None
                freq_1                     NaN          NaN  ...          NaN    0.333333
                freq_2                     NaN          NaN  ...          NaN    0.333333
                freq_3                     NaN          NaN  ...          NaN    0.333333
                freq_4                     NaN          NaN  ...          NaN         NaN
                freq_5                     NaN          NaN  ...          NaN         NaN
                1%                    4.400000     2.200000  ...     0.100000         NaN
                5%                    4.600000     2.345000  ...     0.200000         NaN
                10%                   4.800000     2.500000  ...     0.200000         NaN
                25%                   5.100000     2.800000  ...     0.300000         NaN
                50%                   5.800000     3.000000  ...     1.300000         NaN
                75%                   6.400000     3.300000  ...     1.800000         NaN
                90%                   6.900000     3.610000  ...     2.200000         NaN
                95%                   7.255000     3.800000  ...     2.300000         NaN
                99%                   7.700000     4.151000  ...     2.500000         NaN

                [42 rows x 5 columns]
            ```
        """
        return describe(self.df, **self.para)


class Bivariate(object):
    """
    Bivariate analysis.

    Args:
        df: Input Pandas DataFrame.
        ftypes: Feature type DataFrame from `ftype()`. If `None`, create it internally.
        dtypes: Data type DataFrame from `dtype()` or it's dictionary form. If `None`, create it internally.
        unique_th: Data uniqueness threshold. For example, `unique_th=0.2` means that if the number of unique value
            in a columns is less than or equal to 20% of the number observation, the data is considered as categorical.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 ftypes: pd.DataFrame = None,
                 dtypes: Union[pd.DataFrame, dict] = None,
                 unique_th: float = 0.2,
                 ):
        self.ftypes = ftype(df, dtypes=dtypes, unique_th=unique_th) if ftypes is None else ftypes
        self.dtypes = dtypes
        self.unique_th = unique_th
        self.df = df

    # --- Numerical vs numerical

    def corr(self):
        num_col = self.ftypes.loc[self.ftypes['ftype'].isin([TYPE_CONTINUOUS, TYPE_DISCRETE])].index
        if not num_col.empty:
            df_num = self.df[num_col]
            p_corr = df_num.corr()
            s_corr = df_num.corr('spearman')
            k_corr = df_num.corr('kendall')
            return p_corr, s_corr, k_corr
        else:
            return None

    # --- Categorical vs categorical

    def contingency_table(self, is_pct=False):
        cat_col = self.ftypes.loc[self.ftypes['ftype'].isin([TYPE_CATEGORICAL, TYPE_BINARY])].index
        if not cat_col.empty:
            df_cat = self.df[cat_col]
            cont_tbl = contingency_table_combined(df_cat, is_pct)
            return cont_tbl
        else:
            return None

    def association_table(self):
        cat_col = self.ftypes.loc[self.ftypes['ftype'].isin([TYPE_CATEGORICAL, TYPE_BINARY])].index
        if not cat_col.empty:
            df_cat = df[cat_col]
            cont_tbl = contingency_table_combined(df_cat, False)
            asso_tbl = association(df_cat, cont_tbl)
            return asso_tbl
        else:
            return None

    # TODO: --- Categorical vs numerical


if __name__ == '__main__':

    from pathlib import Path

    # Read data
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "diabetic_data.csv"
    df = pd.read_csv(data_file)

    # TODO: data processing

    # Univariate analysis
    u = Univariate(df)
    u.describe()

    # Bivariate analysis
    b = Bivariate(df)
    b.corr()
    b.contingency_table()
    b.association_table()

    print()
