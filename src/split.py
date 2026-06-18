import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


PATH = Path("../data/source/diabetic_data.csv")
SPLITTED_PATH = Path("../data/interim/splitted")


def split_data(dfr,
               target_column="readmitted",
               test_size_step_1=0.2,
               test_size_step_2=0.2,
               random_state=365):
    """
    Split a DataFrame into stratified training, validation, and test sets.

    The function performs the split in two stages:
    1. Split the full dataset into a training pool and a test set.
    2. Split the training pool into the final training and validation sets.

    Stratification is applied in both stages so each output target vector keeps
    approximately the same class distribution as the original target column.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Full dataset containing both feature columns and the target column.
    target_column : str, default="readmitted"
        Name of the target variable column.
    test_size_step_1 : float, default=0.2
        Fraction of the full dataset assigned to the test set.
    test_size_step_2 : float, default=0.2
        Fraction of the first-stage training pool assigned to the validation set.
        With the default values, the final split is 64% train, 16% validation,
        and 20% test.
    random_state : int, default=365
        Random seed used to make the splits reproducible.

    Returns
    -------
    tuple
        X_train_mini, X_val, X_test, y_train_mini, y_val, y_test
    """

    # Separate the feature columns (X) from the target column (y).
    X = dfr.drop(target_column, axis=1)
    y = dfr[target_column]

    # Step 1: create the holdout test set from the full dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_step_1,
        shuffle=True,
        stratify=y,
        random_state=random_state)

    # Step 2: split the remaining training pool into final train and validation sets.
    X_train_mini, X_val, y_train_mini, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size_step_2,
        shuffle=True,
        stratify=y_train,
        random_state=random_state)

    print("Data split report")
    print("-" * 5)
    print(f"Original data shape: {dfr.shape}")
    print(f"X_train_mini shape: {X_train_mini.shape} | y_train_mini shape: {y_train_mini.shape}")
    print(f"X_val shape:        {X_val.shape} | y_val shape:        {y_val.shape}")
    print(f"X_test shape:       {X_test.shape} | y_test shape:       {y_test.shape}")
    print("-" * 5)
    print(f"Train rows:      {len(y_train_mini)} ({len(y_train_mini) / len(y):.2%} of full data)")
    print(f"Validation rows: {len(y_val)} ({len(y_val) / len(y):.2%} of full data)")
    print(f"Test rows:       {len(y_test)} ({len(y_test) / len(y):.2%} of full data)")

    return X_train_mini, X_val, X_test, y_train_mini, y_val, y_test


def create_y_class_report(y_train_mini, y_val, y_test):
    """
    Create a brief pandas report with class counts and percentages for each Y dataset.

    Parameters
    ----------
    y_train_mini : pandas.Series or array-like
        Target values for the final training set.
    y_val : pandas.Series or array-like
        Target values for the validation set.
    y_test : pandas.Series or array-like
        Target values for the test set.

    Returns
    -------
    pandas.DataFrame
        One row per dataset/class combination with count and percentage columns.
    """
    y_datasets = {
        "Train": pd.Series(y_train_mini, name="target"),
        "Validation": pd.Series(y_val, name="target"),
        "Test": pd.Series(y_test, name="target"),
    }

    # Use one shared class order so the rows are easy to compare across datasets.
    class_order = pd.concat(y_datasets.values()).drop_duplicates().tolist()

    report_rows = []
    for dataset_name, y_values in y_datasets.items():
        total_rows = len(y_values)
        class_counts = y_values.value_counts().reindex(class_order, fill_value=0)
        class_percentages = (class_counts / total_rows * 100).round(2)

        for class_label in class_order:
            report_rows.append({
                "Dataset": dataset_name,
                "Class": class_label,
                "Count": int(class_counts[class_label]),
                "Percentage": class_percentages[class_label],
                "Total Rows": total_rows,
            })

    report = pd.DataFrame(report_rows)

    return report


def export_split_data_to_csv(X_train_mini,
                             X_val,
                             X_test,
                             y_train_mini,
                             y_val,
                             y_test,
                             output_folder=SPLITTED_PATH,
                             target_column="readmitted"):
    """
    Export the split feature and target datasets to CSV files.

    The function creates one CSV file per dataset inside `output_folder`:
    X_train_mini.csv, X_val.csv, X_test.csv,
    y_train_mini.csv, y_val.csv, and y_test.csv.

    Parameters
    ----------
    X_train_mini, X_val, X_test : pandas.DataFrame
        Feature datasets created by `split_data`.
    y_train_mini, y_val, y_test : pandas.Series or array-like
        Target datasets created by `split_data`.
    output_folder : str or pathlib.Path, default=SPLITTED_PATH
        Folder where the CSV files will be saved.
    target_column : str, default="readmitted"
        Column name used when saving the target datasets.

    Returns
    -------
    dict
        Dictionary mapping each dataset name to its exported CSV path.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    feature_datasets = {
        "X_train_mini": X_train_mini,
        "X_val": X_val,
        "X_test": X_test,
    }
    target_datasets = {
        "y_train_mini": y_train_mini,
        "y_val": y_val,
        "y_test": y_test,
    }

    exported_paths = {}

    for dataset_name, dataset in feature_datasets.items():
        file_path = output_folder / f"{dataset_name}.csv"
        pd.DataFrame(dataset).to_csv(file_path, index=False)
        exported_paths[dataset_name] = file_path

    for dataset_name, y_values in target_datasets.items():
        file_path = output_folder / f"{dataset_name}.csv"
        pd.Series(y_values, name=target_column).to_frame().to_csv(file_path, index=False)
        exported_paths[dataset_name] = file_path

    print(f"Exported {len(exported_paths)} split CSV files to: {output_folder}")
    for dataset_name, file_path in exported_paths.items():
        print(f"- {dataset_name}: {file_path}")

    return exported_paths

