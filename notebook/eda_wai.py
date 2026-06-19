import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from pathlib import Path
from src.pipelines import *
from src.split import *
from src.analysis import *


PATH_DATA = Path("../data/source/diabetic_data.csv")
PATH_DATA_DICT = Path("../data/source/data_dictionary.csv")


if __name__ == "__main__":

    # ------------------------------------------------------------------------------------

    ### Data loading and pre-processing

    # Load data and replace '?' with nan
    df_raw = pd.read_csv(PATH_DATA).astype(str).replace('?', np.nan)

    # Load data dictionary
    df_data_dict = pd.read_csv(PATH_DATA_DICT)

    # Get integer / categorical / id feature columns
    col_int = df_data_dict.loc[df_data_dict['Type'] == 'Integer']['Variable Name'].tolist()
    col_cat = df_data_dict.loc[(df_data_dict['Type'] == 'Categorical') & (df_data_dict['Variable Name'] != 'readmitted')]['Variable Name'].tolist()
    col_y   = ['readmitted']
    col_ids = [c for c in df.columns if c not in col_int + col_cat + col_y]

    col_drop = []

    # Convert numerical columns to integers
    df_raw[col_int] = df_raw[col_int].astype(int)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_raw)

    # UNIVARIATE DATA ANALYSIS
    # ------------------------------------------------------------------------------------

    ### Univariate analysis tables

    print('Univariate data analysis summary')

    # Get univariate analysis for integer columns
    u_int = Univariate(X_train[col_int].reset_index(drop=True), unique_th=0)
    eda_u_int = u_int.describe().T

    # Get univariate analysis for categorical columns
    u_cat = Univariate(X_train[col_cat].reset_index(drop=True), unique_th=1)
    eda_u_cat = u_cat.describe().T

    # ------------------------------------------------------------------------------------

    ### Missing data analysis

    print('Missing data analysis')
    print('\n')

    df_missing = pd.concat([eda_u_cat[['missing']], eda_u_int[['missing']]], axis=0, ignore_index=False) / X_train.shape[0]

    # Plot missing
    print('Plot missing percentages:')
    plot_df = df_missing.reset_index()
    plot_df.columns = ["feature", "missing"]
    plot_df = plot_df.sort_values("missing", ascending=False)
    plt.figure(figsize=(10, 12))
    ax = sns.barplot(
        data=plot_df,
        x="missing",
        y="feature",
        color="steelblue"
    )
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set(
        xlabel="Missing values (%)",
        ylabel="Feature",
        title="Percentage of Missing Values by Feature"
    )
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.show()
    print('\n')

    print('Observation:')
    print(f'- Among the input features, '
          f'`weight` ({df_missing.loc['weight']['missing']:.04}), '
          f'`max_glu_serum` ({df_missing.loc['max_glu_serum']['missing']:.04}), '
          f'`A1Cresult` ({df_missing.loc['A1Cresult']['missing']:.04}), '
          f'`medical_specialty` ({df_missing.loc['medical_specialty']['missing']:.04}), and '
          f'`payer_code` ({df_missing.loc['payer_code']['missing']:.04}) contains material proportions of missing values.')
    print('- From the data dictionary, we assume `encounter_id`, `patient_nbr`, `weight` and `payer_code` have no implications to the readmission classification.')
    print('- Despite the relatively high missing percentages, `max_glu_serum` and `A1Cresult` are tests that reflect blood sugar levels and are associated with diabetes severeness.')
    print('- `medical_specialty` may contain symptoms that are related to diabetes.')
    print('\n')

    print('Decision:')
    print('- Drop columns `encounter_id`, `patient_nbr`, `weight` and `payer_code`.')
    print('- Keep `medical_specialty`, `max_glu_serum` and `A1Cresult`.')

    # Drop columns
    col_drop.extend(['encounter_id', 'patient_nbr', 'weight', 'payer_code'])
    col_cat = [c for c in col_cat if c not in col_drop]
    col_int = [c for c in col_int if c not in col_drop]
    X_train.drop(col_drop, axis=1, inplace=True)
    X_test.drop(col_drop, axis=1, inplace=True)
    X_val.drop(col_drop, axis=1, inplace=True)

    # ------------------------------------------------------------------------------------

    ### Cardinality analysis

    print('Cardinality analysis')
    print('\n')

    df_unique = eda_u_cat.loc[col_cat][['nobs', 'distinct']]
    df_unique['unique_count'] = df_unique['distinct']
    df_unique['unique_ratio'] = df_unique['unique_count'] / df_unique['nobs']
    df_unique.sort_values('unique_count', ascending=False, inplace=True)
    df_unique.index.name = 'feature'
    df_unique.reset_index(inplace=True)

    # Plot unique counts and ratios
    print('Plot unique counts and ratios:')
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 10)
    )
    sns.barplot(
        data=df_unique.sort_values("unique_count", ascending=False),
        x="unique_count",
        y="feature",
        ax=axes[0],
        color="steelblue"
    )
    axes[0].set(
        title="Unique Count by Feature",
        xlabel="Number of Unique Values",
        ylabel="Feature"
    )
    sns.barplot(
        data=df_unique.sort_values("unique_ratio", ascending=False),
        x="unique_ratio",
        y="feature",
        ax=axes[1],
        color="darkorange"
    )
    axes[1].set(
        title="Unique Ratio by Feature",
        xlabel="Percentage of Unique Values",
        ylabel=""
    )
    axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------

    ### Target variable analysis

    print('Target variable analysis')
    print('\n')

