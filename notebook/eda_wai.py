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

    ### Data loading and pre-processing

    # Load data and replace '?' with nan
    df = pd.read_csv(PATH_DATA).astype(str).replace('?', np.nan)

    # Load data dictionary
    df_data_dict = pd.read_csv(PATH_DATA_DICT)

    # Get integer / categorical / id feature columns
    col_int = df_data_dict.loc[df_data_dict['Type'] == 'Integer']['Variable Name'].tolist()
    col_cat = df_data_dict.loc[df_data_dict['Type'] == 'Categorical']['Variable Name'].tolist()
    col_ids = [c for c in df.columns if c not in col_int + col_cat]

    # Convert numerical columns to integers
    df[col_int] = df[col_int].astype(int)

    ### Univariate analysis tables

    # Get univariate analysis for integer columns
    u_int = Univariate(df[col_int], unique_th=0)
    eda_u_int = u_int.describe().T

    # Get univariate analysis for categorical columns
    u_cat = Univariate(df[col_cat], unique_th=1)
    eda_u_cat = u_cat.describe().T

    ### Missing data analysis
    df_missing = pd.concat([eda_u_cat[['missing']], eda_u_int[['missing']]], axis=0, ignore_index=False) / df.shape[0]

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
    plt.tight_layout()
    plt.show()

    #