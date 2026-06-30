"""
Automatic Data Fetching Utility Classes

Import this to your notebook:
from src.utils.data_fetch import DataLoader

How to use it:
df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest = DataLoader().fetch_data(version = 'interim')

Other versions: 'final_raw', 'final_label', 'final_onehot'
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



class DataLoader(DataRead):
    """
    Handles data loading operations from various directory structures.

    This class is responsible for managing paths to data files and reading
    them into usable formats. It supports different versions of data
    organization (`interim` and `final_`) and provides an interface to
    extract and load datasets for training, validation, and testing.
    """
    def __init__(self):
        # self. = ".paths"
        super().__init__()
        
    def choose_data(self, version: str = 'interim'):
            # from interim directory -> split_kf_group_stratified
            if version == 'interim':
                # X
                self.X_PATH_TRAIN_MINI_KF_INT = os.getenv("X_PATH_TRAIN_MINI_KF_INT")
                self.X_PATH_VAL_KF_INT= os.getenv("X_PATH_VAL_KF_INT")
                self.X_PATH_TEST_KF_INT = os.getenv("X_PATH_TEST_KF_INT")
                # Y
                self.Y_PATH_TRAIN_MINI_KF_INT = os.getenv("Y_PATH_TRAIN_MINI_KF_INT")
                self.Y_PATH_VAL_KF_INT = os.getenv("Y_PATH_VAL_KF_INT")
                self.Y_PATH_TEST_KF_INT = os.getenv("Y_PATH_TEST_KF_INT")

                # tuple of paths
                return (self.X_PATH_TRAIN_MINI_KF_INT, self.X_PATH_VAL_KF_INT, self.X_PATH_TEST_KF_INT,
                        self.Y_PATH_TRAIN_MINI_KF_INT, self.Y_PATH_VAL_KF_INT, self.Y_PATH_TEST_KF_INT,
                )
            
            elif version == 'final_raw':
                # X
                self.X_PATH_TRAIN_MINI_KF_F_RAW = os.getenv("X_PATH_TRAIN_MINI_KF_F_RAW")
                self.X_PATH_VAL_KF_F_RAW = os.getenv("X_PATH_VAL_KF_F_RAW")
                self.X_PATH_TEST_KF_F_RAW = os.getenv("X_PATH_TEST_KF_F_RAW")
                # Y
                self.Y_PATH_TRAIN_MINI_KF_F_RAW = os.getenv("Y_PATH_TRAIN_MINI_KF_F_RAW")
                self.Y_PATH_VAL_KF_F_RAW = os.getenv("Y_PATH_VAL_KF_F_RAW")
                self.Y_PATH_TEST_KF_F_RAW = os.getenv("Y_PATH_TEST_KF_F_RAW")

                # tuple of paths
                return (self.X_PATH_TRAIN_MINI_KF_F_RAW, self.X_PATH_VAL_KF_F_RAW, self.X_PATH_TEST_KF_F_RAW,
                        self.Y_PATH_TRAIN_MINI_KF_F_RAW, self.Y_PATH_VAL_KF_F_RAW, self.Y_PATH_TEST_KF_F_RAW,
                )

            elif version == 'final_label':
                # X
                self.X_PATH_TRAIN_MINI_KF_F_LE = os.getenv("X_PATH_TRAIN_MINI_KF_F_LE")
                self.X_PATH_VAL_KF_F_LE = os.getenv("X_PATH_VAL_KF_F_LE")
                self.X_PATH_TEST_KF_F_LE = os.getenv("X_PATH_TEST_KF_F_LE")
                # Y
                self.Y_PATH_TRAIN_MINI_KF_F_LE = os.getenv("Y_PATH_TRAIN_MINI_KF_F_LE")
                self.Y_PATH_VAL_KF_F_LE = os.getenv("Y_PATH_VAL_KF_F_LE")
                self.Y_PATH_TEST_KF_F_LE = os.getenv("Y_PATH_TEST_KF_F_LE")

                # tuple of paths
                return (self.X_PATH_TRAIN_MINI_KF_F_LE, self.X_PATH_VAL_KF_F_LE, self.X_PATH_TEST_KF_F_LE,
                        self.Y_PATH_TRAIN_MINI_KF_F_LE, self.Y_PATH_VAL_KF_F_LE, self.Y_PATH_TEST_KF_F_LE,
                )

            elif version == 'final_onehot':
                # X
                self.X_PATH_TRAIN_MINI_KF_F_OH = os.getenv("X_PATH_TRAIN_MINI_KF_F_OH")
                self.X_PATH_VAL_KF_F_OH = os.getenv("X_PATH_VAL_KF_F_OH")
                self.X_PATH_TEST_KF_F_OH = os.getenv("X_PATH_TEST_KF_F_OH")
                # Y
                self.Y_PATH_TRAIN_MINI_KF_F_OH = os.getenv("Y_PATH_TRAIN_MINI_KF_F_OH")
                self.Y_PATH_VAL_KF_F_OH = os.getenv("Y_PATH_VAL_KF_F_OH")
                self.Y_PATH_TEST_KF_F_OH = os.getenv("Y_PATH_TEST_KF_F_OH")

                # tuple of paths
                return (self.X_PATH_TRAIN_MINI_KF_F_OH, self.X_PATH_VAL_KF_F_OH, self.X_PATH_TEST_KF_F_OH,
                        self.Y_PATH_TRAIN_MINI_KF_F_OH, self.Y_PATH_VAL_KF_F_OH, self.Y_PATH_TEST_KF_F_OH,
                )

            else:
                raise ValueError(
                f"""Invalid version: {version}. 
                Choose from:
                'interim', 'final_raw', 'final_label', or 'final_onehot'.
                """)


    def fetch_data(self, version: str = 'interim'):
        #extract data from choose data path
        paths_tuple = self.choose_data(version)

        df_Xtrain = self.data_read(paths_tuple[0])
        df_Xval = self.data_read(paths_tuple[1])
        df_Xtest = self.data_read(paths_tuple[2])
        df_ytrain = self.data_read(paths_tuple[3])
        df_yval = self.data_read(paths_tuple[4])
        df_ytest = self.data_read(paths_tuple[5])

        print(f"""Data imported succesfully from paths""")
        return df_Xtrain, df_Xval, df_Xtest, df_ytrain, df_yval, df_ytest
