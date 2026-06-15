import numpy as np
import pandas as pd
from analysis import Univariate
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def replace_question_marks(X):
    """
    Replace question marks by NaNs in input data X.

    Args:
        X: Input data as either a dataframe or numpy array.

    Returns:
        X: Modified data with question marks replaced by `nan`.
    """
    return X.replace("?", np.nan) if isinstance(X, pd.DataFrame) else np.where(X == "?", np.nan, X)

question_marks_converter = FunctionTransformer(replace_question_marks)
numeric_transformer = Pipeline(
    steps=[
        ("replace_question_marks", question_marks_converter,),
    ]
)
if __name__ == '__main__':

    path_data = r'D:\Projects\datasci_207_project\ds207_final_project\data\source\diabetic_data.csv'
    path_dict = r'D:\Projects\datasci_207_project\ds207_final_project\data\source\data_dictionary.csv'
    target_var = 'readmitted'

    # Load data and dictionary
    df = pd.read_csv(path_data)
    df_data_dict = pd.read_csv(path_dict)

    # Replace '?' with nan
    df = replace_question_marks(df)

    # Get integer and categorical feature columns
    col_int = df_data_dict.loc[df_data_dict['Type'] == 'Integer']['Variable Name'].tolist()
    col_cat = df_data_dict.loc[df_data_dict['Type'] == 'Categorical']['Variable Name'].tolist()

    # Get univariate analysis for integer columns
    u_int = Univariate(df[col_int], unique_th=0)
    eda_u_int = u_int.describe().T

    # Get univariate analysis for categorical columns
    u_cat = Univariate(df[col_cat], unique_th=0.2)
    eda_u_cat = u_cat.describe().T

    #########################################################################

    """
        DATA PRE-PROCESSING FOR CATEGORICAL VARIABLES
        
        race                    : fill nan with n/a; one-hot encoding
        gender                  : one-hot encoding
        age                     : ordinal encoding
        weight                  : drop
        admission_type_id       : grouping + one-hot encoding
        discharge_disposition_id: grouping + one-hot encoding
        admission_source_id     : grouping + one-hot encoding
        payer_code              : drop; or fill nan with n/a; grouping + one-hot encoding
        medical_specialty       : fill nan with n/a; grouping + one-hot encoding (for logistic); embedding (for NN)
        diag_1                  : fill nan with n/a; grouping + one-hot encoding (for logistic); (grouping +) embedding (for NN)
        diag_2                  : fill nan with n/a; grouping + one-hot encoding (for logistic); (grouping +) embedding (for NN)
        diag_3                  : fill nan with n/a; grouping + one-hot encoding (for logistic); (grouping +) embedding (for NN)
        max_glu_serum           : fill nan with n/a; one-hot encoding
        A1Cresult               : fill nan with n/a; one-hot encoding
        metformin               : check association with readmitted + drop if NO > 99%; one-hot encoding
        repaglinide             : check association with readmitted + drop if NO > 99%; one-hot encoding
        nateglinide             : check association with readmitted + drop if NO > 99%; one-hot encoding
        chlorpropamide          : check association with readmitted + drop if NO > 99%; one-hot encoding
        glimepiride             : check association with readmitted + drop if NO > 99%; one-hot encoding
        acetohexamide           : check association with readmitted + drop if NO > 99%; one-hot encoding
        glipizide               : check association with readmitted + drop if NO > 99%; one-hot encoding
        glyburide               : check association with readmitted + drop if NO > 99%; one-hot encoding
        tolbutamide             : check association with readmitted + drop if NO > 99%; one-hot encoding
        pioglitazone            : check association with readmitted + drop if NO > 99%; one-hot encoding
        rosiglitazone           : check association with readmitted + drop if NO > 99%; one-hot encoding
        acarbose                : check association with readmitted + drop if NO > 99%; one-hot encoding
        miglitol                : check association with readmitted + drop if NO > 99%; one-hot encoding
        troglitazone            : check association with readmitted + drop if NO > 99%; one-hot encoding
        tolazamide              : check association with readmitted + drop if NO > 99%; one-hot encoding
        examide                 : drop
        citoglipton             : drop
        insulin                 : check association with readmitted + drop if NO > 99%; one-hot encoding
        glyburide-metformin     : check association with readmitted + drop if NO > 99%; one-hot encoding
        glipizide-metformin     : check association with readmitted + drop if NO > 99%; one-hot encoding
        glimepiride-pioglitazone: check association with readmitted + drop if NO > 99%; one-hot encoding
        metformin-rosiglitazone : check association with readmitted + drop if NO > 99%; one-hot encoding
        metformin-pioglitazone  : check association with readmitted + drop if NO > 99%; one-hot encoding
        change                  : check association with readmitted + drop if NO > 99%; one-hot encoding
        diabetesMed             : check association with readmitted + drop if NO > 99%; one-hot encoding
        readmitted              : binary transformation
        
        had_emergency      = (number_emergency > 0)
        high_comorbidity   = (number_diagnoses >= top_quartile)
        diabetes_dx_count  = diag_1.startswith('250') + diag_2.startswith('250') + diag_3.startswith('250')
    """

    """
        DATA PRE-PROCESSING FOR INTEGER VARIABLES
        
        time_in_hospital        : standardization
        num_lab_procedures      : standardization
        num_procedures          : standardization
        num_medications         : standardization
        number_outpatient       : standardization
        number_emergency        : standardization
        number_inpatient        : standardization
        number_diagnoses        : standardization
    """