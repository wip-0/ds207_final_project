"""
Automatic Data Fetching Utility Classes

Import this to your notebook:
from src.utils.data_fetch import DataLoader

How to use it:
df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest = DataLoader().fetch_data_stratified()
# or
df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest = DataLoader().fetch_data_group_shuffling()

"""

# imports
import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd


class FindProjectRoot:
    """
    Class for identifying the root directory of a project.

    This class provides a method to search for and determine the root directory
    of a project by looking for the presence of a predefined file or marker (`.paths`).
    It is useful in managing file paths or configuration setups within a project.

    :ivar project_root: The identified root directory of the project.
    :type project_root: Path
    """
    def __init__(self, start: Path | None = None):
        self.project_root = self.find_project_root(start)

    def find_project_root(self, start: Path | None = None) -> Path:
        """Find the project root by searching upward for the .paths file."""
        if start is None:
            start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()

        for path in [start, *start.parents]:
            if (path / ".paths").exists():
                return path

        raise FileNotFoundError("Could not find project root containing '.paths' file")


class DataRead(FindProjectRoot):
    """
    Handles data reading functionality and integration with project root determination.

    This class extends the `FindProjectRoot` class and provides a method to read data
    (from file paths) within the determined project structure. It ensures environment
    variables from the `.paths` file are loaded for proper configuration.

    :ivar BASE_DIR: Represents the base directory of the project as determined
        by the `find_project_root` method.
    :type BASE_DIR: Path
    """
    def __init__(self):
        # self. = ".paths"
        super().__init__()

        # IMPORTANT: invoke GitHub Links from .paths file
        # update .paths file with the latest links if necessary
        BASE_DIR = self.find_project_root(start = None)
        load_dotenv(BASE_DIR / ".paths")

    def data_read(self, path_: str | None):
        df = pd.read_csv(path_, low_memory=False)
        print('Data loaded from path: ', path_, '')
        return df



###### Import this to your notebook:
# Example below:
# from src.utils.data_fetch import DataLoader
# df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest = DataLoader().fetch_data_stratified()
# or
# df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest = DataLoader().fetch_data_group_shuffling()

#TODO: The multiple methods may be optimised if needed to one method with different optons

class DataLoader(DataRead):
    """
    Handles the loading of data paths and fetching datasets for
    both classic stratified split and group shuffle split.

    This class provides mechanisms to load file paths for training,
    validation, and testing datasets based on stratified splits or
    group shuffle splits. It is used for structured, efficient, and
    configurable data loading processes.

    :ivar X_PATH_TRAIN_MINI: Path for the stratified mini training dataset.
    :type X_PATH_TRAIN_MINI: str | None
    :ivar X_PATH_VAL: Path for the stratified validation dataset.
    :type X_PATH_VAL: str | None
    :ivar X_PATH_TEST: Path for the stratified test dataset.
    :type X_PATH_TEST: str | None
    :ivar paths_data_: Tuple of all paths (train, validation, test) for the
        stratified split.
    :type paths_data_: tuple[str | None, str | None, str | None]
    :ivar X_PATH_TRAIN_MINI_GS: Path for the group shuffle mini training dataset.
    :type X_PATH_TRAIN_MINI_GS: str | None
    :ivar X_PATH_VAL_GS: Path for the group shuffle validation dataset.
    :type X_PATH_VAL_GS: str | None
    :ivar X_PATH_TEST_GS: Path for the group shuffle test dataset.
    :type X_PATH_TEST_GS: str | None
    :ivar paths_data_gs: Tuple of all paths (train, validation, test) for the
        group shuffle split.
    :type paths_data_gs: tuple[str | None, str | None, str | None]
    """
    def __init__(self):
        # self. = ".paths"
        super().__init__()

        ##################################
        ### classic stratified split paths
        self.X_PATH_TRAIN_MINI = os.getenv("X_PATH_TRAIN_MINI")
        self.X_PATH_VAL = os.getenv("X_PATH_VAL")
        self.X_PATH_TEST = os.getenv("X_PATH_TEST")

        self.Y_PATH_TRAIN_MINI = os.getenv("Y_PATH_TRAIN_MINI")
        self.Y_PATH_VAL = os.getenv("Y_PATH_TRAIN_MINI")
        self.Y_PATH_TEST = os.getenv("Y_PATH_TRAIN_MINI")

        # tuple of paths
        self.paths_data_ = (self.X_PATH_TRAIN_MINI,
                            self.X_PATH_VAL,
                            self.X_PATH_TEST,
                            self.Y_PATH_TRAIN_MINI,
                            self.Y_PATH_VAL,
                            self.Y_PATH_TEST
                            )

        ##################################
        ### group shuffle split paths ###
        self.X_PATH_TRAIN_MINI_GS = os.getenv("X_PATH_TRAIN_MINI_GS")
        self.X_PATH_VAL_GS = os.getenv("X_PATH_VAL_GS")
        self.X_PATH_TEST_GS = os.getenv("X_PATH_TEST_GS")

        self.Y_PATH_TRAIN_MINI_GS = os.getenv("Y_PATH_TRAIN_MINI_GS")
        self.Y_PATH_VAL_GS = os.getenv("Y_PATH_TRAIN_MINI_GS")
        self.Y_PATH_TEST_GS = os.getenv("Y_PATH_TRAIN_MINI_GS")

        # tuple of paths
        self.paths_data_gs = (self.X_PATH_TRAIN_MINI_GS,
                              self.X_PATH_VAL_GS,
                              self.X_PATH_TEST_GS,
                              self.Y_PATH_TRAIN_MINI_GS,
                              self.Y_PATH_VAL_GS,
                              self.Y_PATH_TEST_GS
                              )

    # to deprecate
    def fetch_data_stratified(self):
        # classic stratified split
        paths_tuple = self.paths_data_

        df_Xtrain = self.data_read(paths_tuple[0])
        df_Xval = self.data_read(paths_tuple[1])
        df_Xtest = self.data_read(paths_tuple[2])
        df_ytrain = self.data_read(paths_tuple[3])
        df_yval = self.data_read(paths_tuple[4])
        df_ytest = self.data_read(paths_tuple[5])

        print(f"""Data imported succesfully from paths""")
        return df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest

    # to deprecate
    def fetch_data_group_shuffling(self):
        # group shuffling by
        # Patient ID + stratification
        paths_tuple = self.paths_data_gs

        df_Xtrain = self.data_read(paths_tuple[0])
        df_Xval = self.data_read(paths_tuple[1])
        df_Xtest = self.data_read(paths_tuple[2])
        df_ytrain = self.data_read(paths_tuple[3])
        df_yval = self.data_read(paths_tuple[4])
        df_ytest = self.data_read(paths_tuple[5])

        print(f"""Data imported succesfully from paths""")
        return df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest

