"""
Functions used for data split (creating sets of training, validation, and test data).
Some additional functions are also included (preprocessing and data split analysis).
"""

#### IMPORTS #####
from sklearn.model_selection import (
GroupShuffleSplit,
StratifiedGroupKFold,
train_test_split
)
import pandas as pd
import numpy as np
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


##### PREPROCESSING FUNCTIONS #####

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



def count_patients(df):
    """
    Add a counter column showing how many times each patient appears.

    A counter value of 1 means the patient appears once in the dataset. A value
    greater than 1 identifies rows belonging to patients with multiple
    encounters.
    """
    # checking if the needed columns exist:
    required_columns = {"encounter_id", "patient_nbr"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    # checking the uniqueness of the encounter_id column:
    if df["encounter_id"].duplicated().any():
        raise ValueError("encounter_id values must be unique.")

    df_with_counter = df.copy()
    df_with_counter["counter"] = df_with_counter.groupby("patient_nbr")["patient_nbr"].transform("count")

    return df_with_counter


def randomly_select_from_multi_patients(
        df,
        seed= 42):
    """
    Keep all single-encounter patients
    and randomly select one row per multi-encounter patient.
    """

    df_count = count_patients(df)
    df_count_only_one = df_count[df_count["counter"] == 1]
    df_count_more_than_one = df_count[df_count["counter"] > 1]
    # group by "patient_nbr" and randomly select one row per patient number
    df_count_more_than_one = (
        df_count_more_than_one
        .groupby("patient_nbr", group_keys= False)
        .sample(n= 1, random_state= seed)
    )
    # concatenation of the two dataframes
    df_result = pd.concat([df_count_only_one, df_count_more_than_one])
    # sort the dataframe by index
    df_result = df_result.sort_index()

    return df_result


#### SPLIT FUNCTIONS ####

def split_data(dfr,
               target_column= "readmitted",
               test_size_step_1= 0.2,
               test_size_step_2= 0.2,
               random_state= 365):
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
        X, y,
        test_size= test_size_step_1,
        shuffle=True,
        stratify= y,
        random_state= random_state)

    # Step 2: split X_train/y_train into final train and validation sets.
    X_train_mini, X_val, y_train_mini, y_val = train_test_split(
        X_train,
        y_train,
        test_size=test_size_step_2,
        shuffle= True,
        stratify= y_train,
        random_state= random_state)

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


def split_data_by_patient_stratified(
    dfr,
    target_column="readmitted",
    patient_column="patient_nbr",
    encounter_column="encounter_id",
    remove_death_hospice= True, # remove rows where patients were discharged to death/hospice facilities
    random_state=365,
):
    """
    Stratifies and splits the provided dataset into training, validation, and test sets
    while ensuring that the splits are grouped by patients. The function preserves the
    group structure of the dataset, such that samples from the same patient will not be
    spread across multiple splits.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    # Create a working copy of the input dataframe to avoid modifying the original
    df = dfr.copy()

    # Optionally remove rows where patients were discharged to death/hospice facilities
    # These specific IDs represent discharge dispositions that indicate death or hospice care
    if remove_death_hospice:
        death_hospice_ids = [
            11,  # Expired
            13,  # Hospice / home
            14,  # Hospice / medical facility
            19,  # Expired at home
            20,  # Expired in a medical facility
            21 # Expired, place unknown
        ]
        # create a boolean mask to filter out rows where discharge_disposition_id is in the list
        is_death_or_hospice = df["discharge_disposition_id"].isin(death_hospice_ids)
        # filter out rows where discharge_disposition_id is in the list
        df = df.loc[~is_death_or_hospice].copy()

    # Validate that the target column contains only binary values (0 and 1)
    # This ensures the data is properly prepared for binary classification
    valid_target_values = set(df[target_column].dropna().unique())
    if not valid_target_values.issubset({0, 1}):
        raise ValueError(
            f"{target_column} must already be binary encoded as 0/1. "
            f"Found values: {sorted(valid_target_values)}"
        )

    # Extract the patient grouping column and target variable for stratified splitting
    groups = df[patient_column]
    y = df[target_column]

    # First split: Create test set (20% of data) using stratified group k-fold
    # This ensures patients are not split across train/validation and test sets
    # and maintains class distribution in both splits
    sgkf_test = StratifiedGroupKFold(n_splits= 5, shuffle= True,
                                     random_state= random_state)
    train_val_idx, test_idx = next(sgkf_test.split(df, y, groups))

    # Create the combined train/validation dataframe and the separate test dataframe
    df_train_val = df.iloc[train_val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # Second split: Further split the train/validation data into separate train and validation sets
    # Using 4 splits means validation will be approximately 25% of the train/validation pool
    sgkf_val = StratifiedGroupKFold(n_splits= 4, shuffle= True,
                                    random_state= random_state)

    # Extract indices for the final training and validation sets
    train_idx, val_idx = next(sgkf_val.split(
                            df_train_val,
                            df_train_val[target_column],
                            df_train_val[patient_column],))

    # Create separate dataframes
    # for training and validation sets
    df_train = df_train_val.iloc[train_idx].copy()
    df_val = df_train_val.iloc[val_idx].copy()

    # Separate features
    # from target so readmitted is not exported in both X and y files.
    X_train_plus_val = df_train_val.drop(columns=[target_column])
    X_train_mini = df_train.drop(columns=[target_column])
    X_val = df_val.drop(columns=[target_column])
    X_test = df_test.drop(columns=[target_column])

    return (X_train_plus_val,
            X_train_mini,
            X_val,
            X_test,
            ## target vars for train, val, and test:
            df_train_val[target_column], # y_train_plus_val
            df_train[target_column], # y_train_mini
            df_val[target_column], # y_val
            df_test[target_column] # y_test
            )



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
    # Confirm the split can use both the target and patient grouping columns.
    required_columns = {target_column, patient_column}
    missing_columns = required_columns.difference(dfr.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Build the feature matrix, target vector, and patient groups.
    # patient_column is used for grouping only and is not returned as a feature.
    X = dfr.drop(columns=[target_column, patient_column])
    y = dfr[target_column]
    groups = dfr[patient_column]

    # Step 1: create the train/validation pool and holdout test set.
    # Grouping keeps all encounters from the same patient in one split.
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

    # Step 2: split the train/validation pool into final train and validation sets.
    # The same patient grouping rule is applied again.
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

    # Collect patient IDs for leakage checks between splits.
    train_pool_patients = set(groups_train)
    train_patients = set(groups_train_mini)
    val_patients = set(groups_val)
    test_patients = set(groups_test)

    # Print split sizes and patient-overlap diagnostics.
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

    # Return the train/validation pool, final train set, validation set, and test set.
    return X_train, X_train_mini, X_val, X_test, y_train, y_train_mini, y_val, y_test



##### DATA SPLIT ANALYSIS FUNCTIONS #####

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


#### EXPORT TO CSV FUNCTIONS ####

def export_split_data_to_csv(X_train_plus_val,
                             X_train_mini,
                             X_val,
                             X_test,
                             y_train_plus_val,
                             y_train_mini,
                             y_val,
                             y_test,
                             output_folder= 'PATH',
                             target_column= "readmitted"):
    """
    Export the split feature and target datasets to CSV files.

    The function creates one CSV file per dataset inside `output_folder`:
    X_train_plus_val.csv, X_train_mini.csv, X_val.csv, X_test.csv,
    y_train_plus_val.csv, y_train_mini.csv, y_val.csv, and y_test.csv.

    Parameters
    ----------
    X_train_plus_val, X_train_mini, X_val, X_test : pandas.DataFrame
        Feature datasets created by `split_data` or `split_data_by_patient`.
        X_train_plus_val is the first-stage training pool before validation split.
    y_train_plus_val, y_train_mini, y_val, y_test : pandas.Series or array-like
        Target datasets created by `split_data` or `split_data_by_patient`.
        y_train_plus_val is the first-stage training pool before validation split.
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
        "X_train_plus_val": X_train_plus_val,
        "X_train_mini": X_train_mini,
        "X_val": X_val,
        "X_test": X_test,
    }
    target_datasets = {
        "y_train_plus_val": y_train_plus_val,
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


#### ADDITIONAL DATA SPLIT FUNCTION ####

def split_data_by_patient_hist(dfr,
                               target_column="readmitted",
                               patient_column="patient_nbr",
                               encounter_column="encounter_id",
                               history_column="historical_encounter_count",
                               test_size_step_1=0.2,
                               test_size_step_2=0.25,
                               random_state=365,
                               ):

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

    # Get the indices for data subsets
    idx_train_mini = X_train_mini.index.copy()
    idx_val        = X_val.index.copy()
    idx_test       = X_test.index.copy()

    # Create a dataframe of data subset label (train_mini=0, val=1, test=2)
    df_split_label = pd.DataFrame(columns=['label'], index=dfr.index)
    df_split_label.loc[idx_train_mini] = 0
    df_split_label.loc[idx_val] = 1
    df_split_label.loc[idx_test] = 2

    # Function that get the data subset labels, sort them, and assign back to particular patient with sorted encounter_id
    def assign_label(x):
        labels = df_split_label.loc[x.index]['label'].tolist()
        labels.sort()
        x.sort_values([encounter_column], inplace=True)
        x['split'] = labels
        x['num_hist_encounters'] = list(range(len(x)))
        return x

    # Get the rows with duplicated patient_nbr
    df_patient_dup = dfr[[encounter_column, patient_column]].loc[dfr[patient_column].duplicated(keep=False)]
    df_patient_dup.index.name = 'index'
    df_patient_dup = df_patient_dup.groupby(patient_column).apply(lambda x: assign_label(x))
    df_patient_dup.reset_index(inplace=True)

    # Drop rows with duplicated patient_nbr from original split data
    X_train_mini.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)
    X_val.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)
    X_test.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)
    y_train_mini.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)
    y_val.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)
    y_test.drop(df_patient_dup['index'], axis=0, errors='ignore', inplace=True)

    # Get the process indices of duplicated patients
    idx_mini_dup = df_patient_dup.loc[df_patient_dup['split'] == 0]['index']
    idx_val_dup  = df_patient_dup.loc[df_patient_dup['split'] == 1]['index']
    idx_test_dup = df_patient_dup.loc[df_patient_dup['split'] == 2]['index']

    # Create data subsets for duplicated patients
    X_train_mini_dup = X.loc[idx_mini_dup]
    X_val_dup        = X.loc[idx_val_dup]
    X_test_dup       = X.loc[idx_test_dup]
    y_train_mini_dup = y.loc[idx_mini_dup]
    y_val_dup        = y.loc[idx_val_dup]
    y_test_dup       = y.loc[idx_test_dup]

    # Populate the number of historical encounters
    X_train_mini['num_hist_encounters'] = 0
    X_val['num_hist_encounters'] = 0
    X_test['num_hist_encounters'] = 0
    X_train_mini_dup['num_hist_encounters'] = df_patient_dup.loc[df_patient_dup['split'] == 0, 'num_hist_encounters'].tolist()
    X_val_dup['num_hist_encounters'] = df_patient_dup.loc[df_patient_dup['split'] == 1, 'num_hist_encounters'].tolist()
    X_test_dup['num_hist_encounters'] = df_patient_dup.loc[df_patient_dup['split'] == 2, 'num_hist_encounters'].tolist()

    # Combine original data and new data
    X_train_mini = pd.concat([X_train_mini, X_train_mini_dup], axis = 0, ignore_index = False)
    X_val        = pd.concat([X_val       , X_val_dup]       , axis = 0, ignore_index = False)
    X_test       = pd.concat([X_test      , X_test_dup]      , axis = 0, ignore_index = False)
    y_train_mini = pd.concat([y_train_mini, y_train_mini_dup], axis = 0, ignore_index = False)
    y_val        = pd.concat([y_val       , y_val_dup]       , axis = 0, ignore_index = False)
    y_test       = pd.concat([y_test      , y_test_dup]      , axis = 0, ignore_index = False)

    # Create arrays of shuffled indices
    idx_shuffle_mini = X_train_mini.index.tolist()
    idx_shuffle_val  = X_val.index.tolist()
    idx_shuffle_test = X_test.index.tolist()

    # Shuffle the indices
    np.random.shuffle(idx_shuffle_mini)
    np.random.shuffle(idx_shuffle_val)
    np.random.shuffle(idx_shuffle_test)

    # Shuffle the data
    X_train_mini = X_train_mini.loc[idx_shuffle_mini]
    X_val        = X_val.loc[idx_shuffle_val]
    X_test       = X_test.loc[idx_shuffle_test]
    y_train_mini = y_train_mini.loc[idx_shuffle_mini]
    y_val        = y_val.loc[idx_shuffle_val]
    y_test       = y_test.loc[idx_shuffle_test]

    return X_train_mini, X_val, X_test, y_train_mini, y_val, y_test
