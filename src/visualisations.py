"""
Visualization functions for the 207-project.

Also, there are two correlation analysis functions in this file:
TBD on where to migrate them.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

def plot_triangular_heatmap(dataframe):
    """
    Plots a triangular correlation heatmap using the seaborn library.
    :param dataframe: A pandas DataFrame containing the data for which the correlation heatmap needs to be plotted.
    :return: None
    """
    plt.figure(figsize=(16, 6))
    #plt.grid(visible = False)
    mask = np.triu(np.ones_like(dataframe.corr()))
    heatmap = sns.heatmap(dataframe.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    return heatmap

def create_plots_box_violin(data, features = None , target = 'readmitted'):
    """
    Create box and violin plots for each column in the given data.

    :param data: The dataset to create the plots for.
    :return: None
    """
    if features is None:
        features = data.columns

    for column in features: # or data.columns
        if column != target:  # we don't want to make a plot for the target variable
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Boxplot
            sns.boxplot(x=target, y=column, data=data, ax=axes[0], palette=['#F05941','#BE3144','#872341'], hue = target)
            axes[0].set_title(f'Boxplot of {column}')  # Set title

            # Violin plot
            sns.violinplot(x=target, y=column, data=data, ax=axes[1], palette=['#F05941','#BE3144','#872341'], hue = target)
            axes[1].set_title(f'Violin plot of {column}')  # Set title

            plt.tight_layout()  # For better spacing between subplots
            plt.show()  # Show the plots



def correlation_with_target(data, feature, target = 'readmitted' ,corr_method = 'kendall'):
    """
    Calculate the correlation (method = kendall) between a feature and the target variable.

    :param data: The input DataFrame containing the data.
    :type data: pandas.DataFrame
    :param feature: The name of the feature to calculate correlation with the target variable.
    :type feature: str
    :return: The correlation between the feature and the target variable.
    :rtype: float or str
    """
    # Check if feature exists in DataFrame
    if feature not in data.columns:
        return f"Feature {feature} not found in DataFrame"

    # Prepare a copy of DataFrame to avoid modifying original data
    df = data.copy()

    # Check target column exists
    if target not in df.columns:
        return f"Target column {target} not found in DataFrame"

    # Check if target variable is categorical, if yes then convert it using Label Encoder
    if df[target].dtype == 'object':
        df[target] = LabelEncoder().fit_transform(df[target])

    # Compute correlation
    correlation = df[[feature, target]].corr(method = corr_method).iloc[0, 1]

    return correlation


def find_correlated_features(df, threshold = 0.5, target = 'readmitted',  corre_method = 'kendall'):
    """
    Computes correlation of each feature with target 'readmitted' and keeps features with correlation >= threshold.

    Params:
    df : pandas.DataFrame, The input dataframe
    threshold : Correlation threshold, only correlations >= threshold are kept

    Returns:
    pandas.core.series.Series, Features that have correlation >= threshold with 'readmitted'.
    """
    correlated_features = {}

    # Iterating over each column(feature) in the DataFrame
    for feature in df.columns:
        # Exclude the target variable 'readmitted'
        if feature != target:
            correlation = correlation_with_target(df, feature, corr_method = corre_method)
            # Check if correlation value is numeric (i.e., the function didn't return an error message)
            if isinstance(correlation, (float, int)):
                # Check if the correlation is greater than or equal to the provided threshold
                if abs(correlation) >= threshold:
                    correlated_features[feature] = correlation

    # Converting the dictionary to a pandas Series for better visual output
    return pd.Series(correlated_features).sort_values(ascending=False)