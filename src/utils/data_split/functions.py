""" Functions used for data split (creating sets of training, validation, and test data)."""

#### IMPORTS #####
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import pandas as pd
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


##### FUNCTIONS #####
def encode_readmitted(y):
    """
    Encodes the 'readmitted' column in a given DataFrame using a predefined mapping.

    The function maps values in the 'readmitted' column to numeric representations:
    'NO' is mapped to 0, '>30' is mapped to 1, and '<30' is mapped to 2.

    :param y: A pandas DataFrame containing a 'readmitted' column with categorical values
        to be encoded. The column must contain only the keys supported by the mapping
        dictionary.
    :return: A new pandas DataFrame with the 'readmitted' column encoded as integers
        based on the predefined mapping.
    """
    mapping = {'NO': 0, '>30': 1, '<30': 2}
    y_encoded = y.copy()
    y_encoded['readmitted'] = y_encoded['readmitted'].map(mapping)
    return y_encoded


def binarize_readmitted(y):
    """
    Binarizes and Encodes the 'readmitted' column in a given DataFrame using a predefined mapping.

    The function maps values in the 'readmitted' column to numeric representations:
    'NO' is mapped to 0, '>30' is mapped to 1, and '<30' is mapped to 1.

    :param y: A pandas DataFrame containing a 'readmitted' column with categorical values
        to be encoded. The column must contain only the keys supported by the mapping
        dictionary.
    :return: A new pandas DataFrame with the 'readmitted' column encoded as integers
        based on the predefined mapping.
    """
    mapping = {'NO': 0, '>30': 1, '<30': 1}
    y_encoded = y.copy()
    y_encoded['readmitted'] = y_encoded['readmitted'].map(mapping)
    return y_encoded



def split_data(dfr,
               target_column= "readmitted",
               test_size_step_1= 0.2,
               test_size_step_2= 0.2,
               random_state=365):
    """
    Split a DataFrame into stratified training, validation, and test sets.

    The function performs the split in two stages:
    1. Split the full dataset into X_train/y_train and a test set.
    2. Split X_train/y_train into the final training and validation sets.

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
        X_train, X_train_mini, X_val, X_test,
        y_train, y_train_mini, y_val, y_test

        X_train and y_train are the first-stage training pool before the
        validation split. X_train_mini and y_train_mini are the final training
        set after the validation split.
    """

    # Separate the feature columns (X) from the target column (y).
    X = dfr.drop(target_column, axis=1)
    y = dfr[target_column]

    # Step 1: create X_train/y_train and the holdout test set from the full dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size_step_1,
        shuffle=True,
        stratify=y,
        random_state=random_state)

    # Step 2: split X_train/y_train into final train and validation sets.
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
    print(f"X_train shape:      {X_train.shape} | y_train shape:      {y_train.shape}")
    print(f"X_train_mini shape: {X_train_mini.shape} | y_train_mini shape: {y_train_mini.shape}")
    print(f"X_val shape:        {X_val.shape} | y_val shape:        {y_val.shape}")
    print(f"X_test shape:       {X_test.shape} | y_test shape:       {y_test.shape}")
    print("-" * 5)
    print(f"Train pool rows: {len(y_train)} ({len(y_train) / len(y):.2%} of full data)")
    print(f"Train rows:      {len(y_train_mini)} ({len(y_train_mini) / len(y):.2%} of full data)")
    print(f"Validation rows: {len(y_val)} ({len(y_val) / len(y):.2%} of full data)")
    print(f"Test rows:       {len(y_test)} ({len(y_test) / len(y):.2%} of full data)")

    return X_train, X_train_mini, X_val, X_test, y_train, y_train_mini, y_val, y_test



def split_data_by_patient(dfr,
                          target_column="readmitted",
                          patient_column="patient_nbr",
                          test_size_step_1= 0.2,
                          test_size_step_2= 0.2,
                          random_state= 365):
    """
    Split encounter-level data into training, validation, and test sets by patient.

    Each row is treated as one hospital encounter, so duplicate patient IDs are
    kept. The patient identifier is used only for grouped splitting and is
    removed from the returned feature datasets to avoid using it as a model
    feature.

    The split is performed in two stages:
    1. Split the full dataset into X_train/y_train and a test set.
    2. Split X_train/y_train into the final training and validation sets.

    Grouping is applied in both stages so no patient appears in more than one
    output set.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Full encounter-level dataset containing the target and patient ID columns.
    target_column : str, default="readmitted"
        Name of the target variable column.
    patient_column : str, default="patient_nbr"
        Name of the patient identifier column used for grouped splitting.
    test_size_step_1 : float, default=0.2
        Fraction of patient groups assigned to the test set.
    test_size_step_2 : float, default=0.2
        Fraction of the first-stage training-pool patient groups assigned to the
        validation set. With the default values, the final split is roughly 64%
        train, 16% validation, and 20% test by rows, depending on patient group
        sizes.
    random_state : int, default=365
        Random seed used to make the splits reproducible.

    Returns
    -------
    tuple
        X_train, X_train_mini, X_val, X_test,
        y_train, y_train_mini, y_val, y_test

        X_train and y_train are the first-stage training pool before the
        validation split. X_train_mini and y_train_mini are the final training
        set after the validation split.
    """
    required_columns = {target_column, patient_column}
    missing_columns = required_columns.difference(dfr.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # patient_column is used for grouping only and is not returned as a feature.
    X = dfr.drop(columns=[target_column, patient_column])
    y = dfr[target_column]
    groups = dfr[patient_column]

    # Step 1:
    # create X_train/y_train and the holdout test set, keeping patients grouped together.
    splitter_step_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size_step_1,
        random_state=random_state)

    train_idx, test_idx = next(
        splitter_step_1.split(X, y, groups)
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    # Step 2:
    # split X_train/y_train into final train and validation sets.
    splitter_step_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size_step_2,
        random_state=random_state)

    train_mini_idx, val_idx = next(
        splitter_step_2.split(X_train, y_train, groups_train)
    )

    X_train_mini, X_val = X_train.iloc[train_mini_idx], X_train.iloc[val_idx]
    y_train_mini, y_val = y_train.iloc[train_mini_idx], y_train.iloc[val_idx]
    groups_train_mini = groups_train.iloc[train_mini_idx]
    groups_val = groups_train.iloc[val_idx]

    train_pool_patients = set(groups_train)
    train_patients = set(groups_train_mini)
    val_patients = set(groups_val)
    test_patients = set(groups_test)

    print("Patient-grouped data split report")
    print("-" * 5)
    print(f"Original data shape: {dfr.shape}")
    print(f"X_train shape:      {X_train.shape} | y_train shape:      {y_train.shape}")
    print(f"X_train_mini shape: {X_train_mini.shape} | y_train_mini shape: {y_train_mini.shape}")
    print(f"X_val shape:        {X_val.shape} | y_val shape:        {y_val.shape}")
    print(f"X_test shape:       {X_test.shape} | y_test shape:       {y_test.shape}")
    print("-" * 5)
    print(f"Train pool rows: {len(y_train)} ({len(y_train) / len(y):.2%} of full data)")
    print(f"Train rows:      {len(y_train_mini)} ({len(y_train_mini) / len(y):.2%} of full data)")
    print(f"Validation rows: {len(y_val)} ({len(y_val) / len(y):.2%} of full data)")
    print(f"Test rows:       {len(y_test)} ({len(y_test) / len(y):.2%} of full data)")
    print("-" * 5)
    print(f"Train pool patients: {len(train_pool_patients)}")
    print(f"Train patients:      {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients:       {len(test_patients)}")
    print("-" * 5)
    print(f"Train/validation patient overlap: {len(train_patients.intersection(val_patients))}")
    print(f"Train/test patient overlap:       {len(train_patients.intersection(test_patients))}")
    print(f"Validation/test patient overlap:  {len(val_patients.intersection(test_patients))}")

    return X_train, X_train_mini, X_val, X_test, y_train, y_train_mini, y_val, y_test


def split_data_by_patient_history(dfr,
                                  target_column="readmitted",
                                  patient_column="patient_nbr",
                                  encounter_column="encounter_id",
                                  history_column="historical_encounter_count",
                                  test_size_step_1=0.2,
                                  test_size_step_2=0.2,
                                  random_state=365):
    """
    Split encounter-level data into random-but-ordered train, validation, and test sets.

    This function differs from ``split_data_by_patient`` in one key way: the same
    patient can appear across train, validation, and test, but only with later
    encounters in later splits. Split membership is still assigned randomly, but
    only at valid chronological cut points. For any patient that appears in
    multiple splits:

    - train encounters have smaller ``encounter_id`` values than validation/test
    - validation encounters have smaller ``encounter_id`` values than test

    A feature column is also added that counts how many prior encounters exist for
    the same patient with a smaller ``encounter_id``.

    Parameters
    ----------
    dfr : pandas.DataFrame
        Full encounter-level dataset containing the target, patient ID, and
        encounter ID columns.
    target_column : str, default="readmitted"
        Name of the target variable column.
    patient_column : str, default="patient_nbr"
        Name of the patient identifier column used to define patient histories.
    encounter_column : str, default="encounter_id"
        Name of the encounter identifier column. Smaller values are treated as
        earlier encounters within a patient.
    history_column : str, default="historical_encounter_count"
        Name of the derived feature containing the number of prior encounters for
        the same patient.
    test_size_step_1 : float, default=0.2
        Approximate fraction of rows assigned to the test set.
    test_size_step_2 : float, default=0.2
        Approximate fraction of the non-test rows assigned to the validation set.
        With the default values, the target split is roughly 64% train, 16%
        validation, and 20% test, but exact ratios are not enforced because
        patient timelines are split using random chronological cut points.
    random_state : int, default=365
        Random seed used to make the random chronological cut points reproducible.

    Returns
    -------
    tuple
        X_train, X_train_mini, X_val, X_test,
        y_train, y_train_mini, y_val, y_test

        ``X_train`` and ``y_train`` are the combined train+validation pool
        produced by the first split stage, consistent with the existing helpers.
    """
    required_columns = {target_column, patient_column, encounter_column}
    missing_columns = required_columns.difference(dfr.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if not 0 < test_size_step_1 < 1:
        raise ValueError("test_size_step_1 must be between 0 and 1.")

    if not 0 < test_size_step_2 < 1:
        raise ValueError("test_size_step_2 must be between 0 and 1.")

    rng = np.random.default_rng(random_state)

    df_sorted = dfr.copy().sort_values(
        by=[patient_column, encounter_column],
        kind="mergesort"
    )

    # Within each patient, the chronological rank is exactly the number of
    # earlier encounters because encounter_id is assumed unique per visit.
    df_sorted[history_column] = df_sorted.groupby(patient_column).cumcount()

    split_labels = pd.Series(index=df_sorted.index, dtype="object")

    for _, patient_df in df_sorted.groupby(patient_column, sort=False):
        patient_indices = patient_df.index.to_numpy()
        patient_size = len(patient_indices)

        # Randomly choose up to two chronological cut points, then assign the
        # resulting contiguous blocks to train -> validation -> test.
        if patient_size == 1:
            train_end = 1
            val_end = 1
        else:
            train_cut = int(rng.integers(1, patient_size + 1))
            val_cut = int(rng.integers(train_cut, patient_size + 1))
            train_end = train_cut
            val_end = val_cut

        split_labels.loc[patient_indices[:train_end]] = "train"
        split_labels.loc[patient_indices[train_end:val_end]] = "val"
        split_labels.loc[patient_indices[val_end:]] = "test"

    train_pool_mask = split_labels.isin(["train", "val"])
    train_mask = split_labels.eq("train")
    val_mask = split_labels.eq("val")
    test_mask = split_labels.eq("test")

    feature_df = df_sorted.drop(columns=[target_column])
    target_df = df_sorted[target_column]

    X_train = feature_df.loc[train_pool_mask]
    X_train_mini = feature_df.loc[train_mask]
    X_val = feature_df.loc[val_mask]
    X_test = feature_df.loc[test_mask]

    y_train = target_df.loc[train_pool_mask]
    y_train_mini = target_df.loc[train_mask]
    y_val = target_df.loc[val_mask]
    y_test = target_df.loc[test_mask]

    patient_splits = pd.DataFrame({
        patient_column: df_sorted[patient_column],
        encounter_column: df_sorted[encounter_column],
        "split": split_labels,
    })
    train_max = patient_splits.loc[train_mask].groupby(patient_column)[encounter_column].max()
    val_min = patient_splits.loc[val_mask].groupby(patient_column)[encounter_column].min()
    val_max = patient_splits.loc[val_mask].groupby(patient_column)[encounter_column].max()
    test_min = patient_splits.loc[test_mask].groupby(patient_column)[encounter_column].min()

    train_val_violations = (
        train_max.to_frame("train_max")
        .join(val_min.rename("val_min"), how="inner")
        .query("train_max >= val_min")
    )
    train_test_violations = (
        train_max.to_frame("train_max")
        .join(test_min.rename("test_min"), how="inner")
        .query("train_max >= test_min")
    )
    val_test_violations = (
        val_max.to_frame("val_max")
        .join(test_min.rename("test_min"), how="inner")
        .query("val_max >= test_min")
    )

    print("Patient-history chronological split report")
    print("-" * 5)
    print(f"Original data shape: {dfr.shape}")
    print(f"X_train shape:      {X_train.shape} | y_train shape:      {y_train.shape}")
    print(f"X_train_mini shape: {X_train_mini.shape} | y_train_mini shape: {y_train_mini.shape}")
    print(f"X_val shape:        {X_val.shape} | y_val shape:        {y_val.shape}")
    print(f"X_test shape:       {X_test.shape} | y_test shape:       {y_test.shape}")
    print("-" * 5)
    print(f"Train pool rows: {len(y_train)} ({len(y_train) / len(target_df):.2%} of full data)")
    print(f"Train rows:      {len(y_train_mini)} ({len(y_train_mini) / len(target_df):.2%} of full data)")
    print(f"Validation rows: {len(y_val)} ({len(y_val) / len(target_df):.2%} of full data)")
    print(f"Test rows:       {len(y_test)} ({len(y_test) / len(target_df):.2%} of full data)")
    print("-" * 5)
    print(f"Train/validation chronology violations: {len(train_val_violations)}")
    print(f"Train/test chronology violations:       {len(train_test_violations)}")
    print(f"Validation/test chronology violations:  {len(val_test_violations)}")

    return X_train, X_train_mini, X_val, X_test, y_train, y_train_mini, y_val, y_test




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



def export_split_data_to_csv(X_train_for_cv,
                             X_train_mini,
                             X_val,
                             X_test,
                             y_train_for_cv,
                             y_train_mini,
                             y_val,
                             y_test,
                             output_folder= 'PATH',
                             target_column= "readmitted"):
    """
    Export the split feature and target datasets to CSV files.

    The function creates one CSV file per dataset inside `output_folder`:
    X_train_for_cv.csv, X_train_mini.csv, X_val.csv, X_test.csv,
    y_train_for_cv.csv, y_train_mini.csv, y_val.csv, and y_test.csv.

    Parameters
    ----------
    X_train_for_cv, X_train_mini, X_val, X_test : pandas.DataFrame
        Feature datasets created by `split_data` or `split_data_by_patient`.
        X_train_for_cv is the first-stage training pool before validation split.
    y_train_for_cv, y_train_mini, y_val, y_test : pandas.Series or array-like
        Target datasets created by `split_data` or `split_data_by_patient`.
        y_train_for_cv is the first-stage training pool before validation split.
    output_folder : str or pathlib.Path, default="PATH"
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
        "X_train_for_cv": X_train_for_cv,
        "X_train_mini": X_train_mini,
        "X_val": X_val,
        "X_test": X_test,
    }
    target_datasets = {
        "y_train_for_cv": y_train_for_cv,
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
