"""
Visualization functions for Analysis of data.

Also, there are two correlation analysis functions in this file:
TBD on where to migrate them.
"""

####### IMPORTS #######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

####### FUNCTIONS #######

def plot_triangular_heatmap(heatmap_df,
                            title= "Correlation Heatmap",
                            color_map= 'BrBG',
                            method='pearson',
                            encode_categoricals=False,
                            ordinal_mappings=None):
    """
    Plot a triangular correlation heatmap.

    Parameters:
        heatmap_df: pandas.DataFrame
            Dataframe containing the variables to correlate.
        title: str
            Title of the heatmap.
        color_map: str
            Colormap used for the heatmap.
        method: str
            Correlation method to use.
            Options: 'pearson', 'spearman', or 'kendall'.
        encode_categoricals: bool
            If True, categorical variables are converted to numeric codes.
        ordinal_mappings: dict, optional
            Dictionary used to manually encode ordinal categorical variables.
            Example:
            {'severity': {'low': 1, 'medium': 2, 'high': 3} }
    Returns:
        seaborn heatmap object.
    """

    # Make a copy so the original dataframe is not modified.
    corr_df = heatmap_df.copy()
    # Apply user-defined ordinal mappings, if provided.
    if ordinal_mappings is not None:
        for column, mapping in ordinal_mappings.items():
            corr_df[column] = corr_df[column].map(mapping)

    # Convert categorical variables to numeric codes if requested.
    if encode_categoricals:
        categorical_columns = corr_df.select_dtypes(['object', 'category']).columns

        for column in categorical_columns:
            corr_df[column] = corr_df[column].astype('category').cat.codes

            # Replace -1 codes, which represent missing values, with NaN.
            corr_df[column] = corr_df[column].replace(-1, np.nan)

    # Calculate correlation matrix using the selected method.
    correlation_matrix = corr_df.corr(method=method)
    # Set the figure size.
    plt.figure(figsize=(16, 6))
    # Create a mask to hide the upper triangle of the correlation matrix.
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    # Plot the lower-triangular correlation heatmap.
    heatmap = sns.heatmap(  correlation_matrix,
                            mask=mask,
                            vmin=-1,
                            vmax=1,
                            annot=True,
                            cmap=color_map
                         )
    # Set heatmap title
    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)

    return heatmap

def plot_correlation_column_heatmap(heatmap_df,
                                    chosen_variable,
                                    title= None,
                                    color_map= 'BrBG',
                                    method= 'pearson',
                                    encode_categoricals= False,
                                    ordinal_mappings= None,
                                    sort_values= False,
                                    exclude_self= False):
    """
    Plot a one-column correlation heatmap for one chosen variable against all variables.

    Parameters:
        heatmap_df: pandas.DataFrame
            Dataframe containing the variables to correlate.
        chosen_variable: str
            The variable to place on the x-axis.
        title: str, optional
            Title of the heatmap.
        color_map: str
            Colormap used for the heatmap.
        method: str
            Correlation method to use.
            Options: 'pearson', 'spearman', or 'kendall'.
        encode_categoricals: bool
            If True, categorical variables are converted to numeric codes.
        ordinal_mappings: dict, optional
            Dictionary used to manually encode ordinal categorical variables.
            Example:
            {'severity': {'low': 1, 'medium': 2, 'high': 3}}
        sort_values: bool
            If True, sort variables by correlation with the chosen variable.
        exclude_self: bool
            If True, remove the chosen variable's correlation with itself.
    Returns:
        seaborn heatmap object.
    """

    # Make a copy so the original dataframe is not modified.
    corr_df = heatmap_df.copy()
    # Apply user-defined ordinal mappings, if provided.
    if ordinal_mappings is not None:
        for column, mapping in ordinal_mappings.items():
            corr_df[column] = corr_df[column].map(mapping)
    # Convert categorical variables to numeric codes if requested.
    if encode_categoricals:
        categorical_columns = corr_df.select_dtypes(['object', 'category']).columns

        for column in categorical_columns:
            corr_df[column] = corr_df[column].astype('category').cat.codes
            # Replace -1 codes, which represent missing values, with NaN.
            corr_df[column] = corr_df[column].replace(-1, np.nan)

    # Keep only numeric columns, because correlation requires numeric values.
    corr_df = corr_df.select_dtypes(include='number')
    # Check that the chosen variable is available after preprocessing.
    if chosen_variable not in corr_df.columns:
        raise ValueError(
            f"'{chosen_variable}' is not numeric or was not found in the dataframe. "
            "Use encode_categoricals=True or provide ordinal_mappings if it is categorical."
        )

    # Calculate the full correlation matrix using the selected method.
    correlation_matrix = corr_df.corr(method=method)
    # Select only the column corresponding to the chosen variable.
    correlation_column = correlation_matrix[[chosen_variable]]
    # Optionally remove the chosen variable's correlation with itself.
    if exclude_self:
        correlation_column = correlation_column.drop(index=chosen_variable)
    # Optionally sort correlations from highest to lowest.
    if sort_values:
        correlation_column = correlation_column.sort_values(by=chosen_variable,
                                                            ascending=False)

    # Set a dynamic figure height based on the number of variables.
    plt.figure(figsize=(4, max(6, 0.35 * len(correlation_column))))
    # Plot the one-column correlation heatmap.
    heatmap = sns.heatmap(correlation_column,
                            vmin=-1,
                            vmax=1,
                            annot=True,
                            cmap=color_map
                          )
    # Set title.
    if title is None:
        title = f'Correlation with {chosen_variable}'

    heatmap.set_title(title, fontdict={'fontsize': 18}, pad=16)
    # Improve axis labels.
    # plt.xlabel(chosen_variable)
    plt.ylabel('Variables')
    plt.tight_layout()
    return heatmap


def plot_categorical_stacked(df,
                             target='readmitted',
                             excluded_feats= None,# example: ['name_feat_1', 'name_feat_2']
                             choose_specific_feats= None,# example: ['name_feat_1', 'name_feat_2']
                             percentage=False,
                             color_map=['#F05941', '#BE3144', '#872341']):
    """
    Plot stacked bar plot/s of categorical variable/s against a target variable.
    Parameters:
        df: pandas.DataFrame
            The dataframe containing the data.
        target: str
            The target variable to compare each categorical feature against.
        excluded_feats: list, optional
            List of features to exclude from the plots.
        choose_specific_feats: list, optional
            List of specific categorical features to plot.
        percentage: bool
            If True, plot percentages/proportions.
            If False, plot raw counts.
        color_map: list
            List of colors used for the stacked bars.
    Returns:
        None
    """

    # Create a custom color map from the given list of colors.
    colors = color_map
    my_cmap = ListedColormap(colors, name="my_cmap")
    # Select categorical columns depending on the user's input.
    if excluded_feats is not None:
        # Drop excluded features, then keep only categorical columns.
        categorical_columns = (
            df.drop(excluded_feats, axis=1)
              .select_dtypes(['object', 'category'])
              .columns
              .tolist()
        )
    elif choose_specific_feats is not None:
        # Use only the specific features chosen by the user.
        categorical_columns = choose_specific_feats
    else:
        # If no feature selection is given, use all categorical columns.
        categorical_columns = (
            df.select_dtypes(['object', 'category'])
              .columns
              .tolist()
        )

    # Remove the target variable from the list of features to plot.
    if target in categorical_columns:
        categorical_columns.remove(target)
        # Plot percentages instead of raw counts.
        if percentage == True:
            for column in categorical_columns:

                # Count observations for each combination of target and feature value.
                counts = df.groupby([target, column]).size().unstack(target)
                # Convert counts into percentages within each feature category.
                counts = counts.apply(lambda x: (x / x.sum()) * 100, axis=1)

                # Plot stacked bar chart.
                counts.plot(
                    kind='bar',
                    stacked=True,
                    figsize=(10, 6),
                    colormap=my_cmap
                )
                # Add title and labels.
                plt.title(f'Stacked bar plot of proportions - {column} per {target} class')
                plt.ylabel('Proportion (%)')
                plt.xticks(rotation=45)
                # Adjust layout and display plot.
                plt.tight_layout()
                plt.show()

        else:
            for column in categorical_columns:

                # Count observations for each combination of target and feature value.
                counts = df.groupby([target, column]).size().unstack(target)
                # Plot stacked bar chart using raw counts.
                counts.plot(
                    kind='bar',
                    stacked=True,
                    figsize=(10, 6),
                    colormap=my_cmap
                )
                # Add title and labels.
                plt.title(f'Stacked bar plot - {column} per {target} class')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                # Adjust layout and display plot.
                plt.tight_layout()
                plt.show()


def create_plots_box_violin(data, features = None ,
                            target = 'readmitted',
                            color_plt = ['#F05941','#BE3144','#872341']):
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
            sns.boxplot(x=target, y=column, data=data, ax=axes[0], palette= color_plt, hue = target)
            axes[0].set_title(f'Boxplot of {column}')  # Set title

            # Violin plot
            sns.violinplot(x=target, y=column, data=data, ax=axes[1], palette= color_plt, hue = target)
            axes[1].set_title(f'Violin plot of {column}')  # Set title

            plt.tight_layout()  # For better spacing between subplots
            plt.show()  # Show the plots



########
### TBD: WHERE TO PLACE THE FUNCTIONS BELOW

def correlation_with_target(data, feature, target = 'readmitted' ,corr_method = 'kendall'):
    """
    Calculate the correlation (method = kendall) between a feature and the target variable.

    :param corr_method: Correlation method to use (defaults to 'kendall').
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

