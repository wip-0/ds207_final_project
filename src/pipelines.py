'''
    Reference: Data Pre-Processing Guide (Original Draft)

    ================================================================================================================

    DATA PRE-PROCESSING FOR CATEGORICAL VARIABLES

    encounter_id            : drop
    patient_nbr             : drop
    race                    : fill nan with n/a; one-hot encoding
    gender                  : one-hot encoding
    age                     : one-hot encoding; ordinal_encoding for NN
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
    examide                 : check association with readmitted + drop if NO > 99%; one-hot encoding
    citoglipton             : check association with readmitted + drop if NO > 99%; one-hot encoding
    insulin                 : check association with readmitted + drop if NO > 99%; one-hot encoding
    glyburide-metformin     : check association with readmitted + drop if NO > 99%; one-hot encoding
    glipizide-metformin     : check association with readmitted + drop if NO > 99%; one-hot encoding
    glimepiride-pioglitazone: check association with readmitted + drop if NO > 99%; one-hot encoding
    metformin-rosiglitazone : check association with readmitted + drop if NO > 99%; one-hot encoding
    metformin-pioglitazone  : check association with readmitted + drop if NO > 99%; one-hot encoding
    change                  : check association with readmitted + drop if NO > 99%; one-hot encoding
    diabetesMed             : check association with readmitted + drop if NO > 99%; one-hot encoding

    ================================================================================================================

    DATA PRE-PROCESSING FOR INTEGER VARIABLES

    time_in_hospital        : standardization
    num_lab_procedures      : standardization
    num_procedures          : standardization
    num_medications         : standardization
    number_outpatient       : standardization
    number_emergency        : standardization
    number_inpatient        : standardization
    number_diagnoses        : standardization

    ================================================================================================================

    DATA PRE-PROCESSING FOR TARGET VARIABLES

    readmitted              : binary transformation

    ================================================================================================================

'''


import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    MaxAbsScaler,
    RobustScaler,
    OneHotEncoder,
    TargetEncoder,
    PowerTransformer,
    QuantileTransformer,
    Normalizer,
    LabelEncoder,
    LabelBinarizer,
    MultiLabelBinarizer,
    OrdinalEncoder,
    KBinsDiscretizer,
    FunctionTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.data_split.functions import binarize_readmitted


LABEL_UNKNOWN = 'Unknown'
PATH_DATA_SHUFFLE = Path('../data/interim/split_groupshuffle')
PATH_DATA_STRATIFY = Path('../data/interim/split_stratified')
PATH_DATA_KF = Path('../data/interim/split_kf_group_stratified')


class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns
        self.is_fitted_ = False

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            return X.iloc[:, 0:0] if isinstance(X, pd.DataFrame) else X[:, 0:0]

        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.columns, errors='ignore')

        idx_drop = set(self.columns)
        idx_keep = [idx for idx in range(X.shape[1]) if idx not in idx_drop]
        return X[:, idx_keep]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', None)

        if input_features is None:
            return np.array([])

        input_features = np.asarray(input_features)
        if self.columns is None:
            return np.array([])

        return np.array([
            feature for idx, feature in enumerate(input_features)
            if feature not in self.columns and idx not in self.columns
        ])


class MedicalSpecialtyGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns or ['medical_specialty']
        self.is_fitted_ = False
        self.mapping_dict = {

            # Missing / Unknown
            '?'                                   : LABEL_UNKNOWN,
            'PhysicianNotFound'                   : LABEL_UNKNOWN,

            # General Medicine & Primary Care
            'InternalMedicine'                    : 'General_Medicine',
            'Family/GeneralPractice'              : 'General_Medicine',
            'Hospitalist'                         : 'General_Medicine',
            'Resident'                            : 'General_Medicine',
            'Osteopath'                           : 'General_Medicine',

            # Emergency
            'Emergency/Trauma'                    : 'Emergency_Acute',

            # Cardiology & Vascular
            'Cardiology'                          : 'Cardiology_Vascular',
            'Surgery-Cardiovascular/Thoracic'     : 'Cardiology_Vascular',
            'Surgery-Vascular'                    : 'Cardiology_Vascular',
            'Surgery-Cardiovascular'              : 'Cardiology_Vascular',
            'Surgery-Thoracic'                    : 'Cardiology_Vascular',

            # Endocrine, Renal & Internal Specialties
            'Nephrology'                          : 'Endocrine_Renal_Internal',
            'Pulmonology'                         : 'Endocrine_Renal_Internal',
            'Gastroenterology'                    : 'Endocrine_Renal_Internal',
            'Oncology'                            : 'Endocrine_Renal_Internal',
            'Hematology/Oncology'                 : 'Endocrine_Renal_Internal',
            'Neurology'                           : 'Endocrine_Renal_Internal',
            'Endocrinology'                       : 'Endocrine_Renal_Internal',
            'Hematology'                          : 'Endocrine_Renal_Internal',
            'InfectiousDiseases'                  : 'Endocrine_Renal_Internal',
            'Rheumatology'                        : 'Endocrine_Renal_Internal',
            'Pathology'                           : 'Endocrine_Renal_Internal',
            'Endocrinology-Metabolism'            : 'Endocrine_Renal_Internal',
            'AllergyandImmunology'                : 'Endocrine_Renal_Internal',
            'Neurophysiology'                     : 'Endocrine_Renal_Internal',

            # General & Specialized Surgery
            'Surgery-General'                     : 'Surgery',
            'Surgeon'                             : 'Surgery',
            'SurgicalSpecialty'                   : 'Surgery',
            'Surgery-Neuro'                       : 'Surgery',
            'Otolaryngology'                      : 'Surgery',
            'Surgery-Plastic'                     : 'Surgery',
            'Surgery-Colon&Rectal'                : 'Surgery',
            'Surgery-Maxillofacial'               : 'Surgery',
            'Surgery-Pediatric'                   : 'Surgery',
            'Surgery-PlasticwithinHeadandNeck'    : 'Surgery',
            'Proctology'                          : 'Surgery',

            # Orthopedics & Rehab (Crucial for Diabetic Podiatry)
            'Orthopedics'                         : 'Orthopedics_Rehab',
            'Orthopedics-Reconstructive'          : 'Orthopedics_Rehab',
            'PhysicalMedicineandRehabilitation'   : 'Orthopedics_Rehab',
            'Podiatry'                            : 'Orthopedics_Rehab',
            'SportsMedicine'                      : 'Orthopedics_Rehab',

            # Radiology
            'Radiologist'                         : 'Radiology_Imaging',
            'Radiology'                           : 'Radiology_Imaging',

            # Mental Health
            'Psychiatry'                          : 'Mental_Health',
            'Psychology'                          : 'Mental_Health',
            'Psychiatry-Child/Adolescent'         : 'Mental_Health',
            'Psychiatry-Addictive'                : 'Mental_Health',
            'Speech'                              : 'Mental_Health',

            # OB/GYN
            'ObstetricsandGynecology'             : 'OB_GYN',
            'Gynecology'                          : 'OB_GYN',
            'Obsterics&Gynecology-GynecologicOnco': 'OB_GYN',
            'Obstetrics'                          : 'OB_GYN',
            'Perinatology'                        : 'OB_GYN',

            # Pediatrics
            'Pediatrics'                          : 'Pediatrics',
            'Pediatrics-Endocrinology'            : 'Pediatrics',
            'Pediatrics-CriticalCare'             : 'Pediatrics',
            'Pediatrics-Pulmonology'              : 'Pediatrics',
            'Anesthesiology-Pediatric'            : 'Pediatrics',
            'Pediatrics-Neurology'                : 'Pediatrics',
            'Cardiology-Pediatric'                : 'Pediatrics',
            'Pediatrics-Hematology-Oncology'      : 'Pediatrics',
            'Pediatrics-EmergencyMedicine'        : 'Pediatrics',
            'Pediatrics-AllergyandImmunology'     : 'Pediatrics',
            'Pediatrics-InfectiousDiseases'       : 'Pediatrics',

            # Other Minor Support Services
            'Urology'                             : 'Other',
            'Ophthalmology'                       : 'Other',
            'Anesthesiology'                      : 'Other',
            'OutreachServices'                    : 'Other',
            'DCPTEAM'                             : 'Other',
            'Dentistry'                           : 'Other',
            'Dermatology'                         : 'Other',
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_dict).fillna(LABEL_UNKNOWN)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.columns)


class AdmissionTypeIdGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns or ['admission_type_id']
        self.is_fitted_ = False
        self.mapping_dict = {
            '1': 'Emergency_Acute',       # Emergency
            '2': 'Emergency_Acute',       # Urgent
            '3': 'Elective_Planned',      # Elective
            '4': 'Other_LowVolume',       # Newborn
            '5': LABEL_UNKNOWN,           # Not Available
            '6': LABEL_UNKNOWN,           # NULL
            '7': 'Emergency_Acute',       # Trauma Center
            '8': LABEL_UNKNOWN,           # Not Mapped
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_dict).fillna(LABEL_UNKNOWN)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.columns)


class AdmissionSourceIdGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns or ['admission_source_id']
        self.is_fitted_ = False
        self.mapping_dict = {
            # Emergency Room
            '7' : 'Emergency_Room',

            # Referrals
            '1' : 'Referral',
            '2' : 'Referral',
            '3' : 'Referral',

            # Transfers from other medical frameworks
            '4' : 'Transfer_From_Facility',
            '5' : 'Transfer_From_Facility',
            '6' : 'Transfer_From_Facility',
            '10': 'Transfer_From_Facility',
            '22': 'Transfer_From_Facility',
            '25': 'Transfer_From_Facility',

            # Missing data flags or rare legal/court admissions
            '8' : LABEL_UNKNOWN,
            '9' : LABEL_UNKNOWN,
            '11': LABEL_UNKNOWN,
            '13': LABEL_UNKNOWN,
            '14': LABEL_UNKNOWN,
            '17': LABEL_UNKNOWN,
            '20': LABEL_UNKNOWN,
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_dict).fillna(LABEL_UNKNOWN)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.columns)


class DischargeDispositionIdGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):

        self.columns = columns or ['discharge_disposition_id']
        self.is_fitted_ = False
        self.mapping_dict = {
            # Discharged Home
            '1': 'Home',
            '6': 'Home',
            '8': 'Home',
            '13': 'Home',
            '14': 'Home',

            # Transferred to alternative healthcare facilities
            '2' : 'Transferred_Facility',
            '3' : 'Transferred_Facility',
            '4' : 'Transferred_Facility',
            '5' : 'Transferred_Facility',
            '22': 'Transferred_Facility',
            '23': 'Transferred_Facility',
            '24': 'Transferred_Facility',
            '28': 'Transferred_Facility',

            # Left Against Medical Advice / Missing / Other
            '7' : LABEL_UNKNOWN,
            '9' : LABEL_UNKNOWN,
            '10': LABEL_UNKNOWN,
            '12': LABEL_UNKNOWN,
            '15': LABEL_UNKNOWN,
            '16': LABEL_UNKNOWN,
            '17': LABEL_UNKNOWN,
            '18': LABEL_UNKNOWN,
            '25': LABEL_UNKNOWN,
            '27': LABEL_UNKNOWN,

            # Patient expired or is in hospice (cannot be readmitted)
            '11': 'Expired_Hospice',
            '19': 'Expired_Hospice',
            '20': 'Expired_Hospice',
        }

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self.mapping_dict).fillna(LABEL_UNKNOWN)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.columns)


class ICD9Grouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns or ['diag_1', 'diag_2', 'diag_3']
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def _convert_code(self, code):
        code = str(code).strip()

        # Handle structural missing or empty entries safely
        if code in ['?', '', 'nan', 'None']:
            return LABEL_UNKNOWN

        if code.startswith('V'):
            return 'Supplementary'
        if code.startswith('E'):
            return 'External_Cause'
        try:
            num = float(code)
        except ValueError:
            return LABEL_UNKNOWN

        if 390 <= num <= 459 or num == 785:
            return 'Circulatory'
        elif 460 <= num <= 519 or num == 786:
            return 'Respiratory'
        elif 520 <= num <= 579 or num == 787:
            return 'Digestive'
        elif 250 <= num < 251:
            return 'Diabetes'
        elif 800 <= num <= 999:
            return 'Injury'
        elif 710 <= num <= 739:
            return 'Musculoskeletal'
        elif 580 <= num <= 629 or num == 788:
            return 'Genitourinary'
        elif 140 <= num <= 239:
            return 'Neoplasms'
        else:
            return 'Other'

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].astype(str).apply(self._convert_code)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else np.array(self.columns)


class HighNoDropper(BaseEstimator, TransformerMixin):

    def __init__(self, th=0.99, val_no='No'):
        self.th = th
        self.val_no = val_no
        self.col_keep = []
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns)
        self.col_keep = X.columns[(((X == 'No').sum() / len(X)) < self.th) == True]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if len(self.col_keep) == 0:
            return pd.DataFrame(index=X.index)
        return X[self.col_keep]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.col_keep)


class PrescriptionGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, handle_unknown='ignore', sparse_output=False):
        self.columns = columns or [
            'metformin'              , 'repaglinide'           , 'nateglinide',
            'chlorpropamide'         , 'glimepiride'           , 'acetohexamide',
            'glipizide'              , 'glyburide'             , 'tolbutamide',
            'pioglitazone'           , 'rosiglitazone'         , 'acarbose',
            'miglitol'               , 'troglitazone'          , 'tolazamide',
            'examide'                , 'citoglipton'           , 'insulin',

            'glyburide-metformin'    , 'glipizide-metformin'   , 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone',
        ]
        self.col_combo = {
            'glyburide-metformin'     : ['glyburide'  , 'metformin'],
            'glipizide-metformin'     : ['glipizide'  , 'metformin'],
            'metformin-rosiglitazone' : ['metformin'  , 'rosiglitazone'],
            'metformin-pioglitazone'  : ['metformin'  , 'pioglitazone'],
            'glimepiride-pioglitazone': ['glimepiride', 'pioglitazone'],
        }
        self.groups = {
            'sulfonylureas'    : ['glimepiride' , 'glipizide'    , 'glyburide'    , 'chlorpropamide', 'acetohexamide', 'tolbutamide', 'tolazamide'],
            'tzds'             : ['pioglitazone', 'rosiglitazone', 'troglitazone'],
            'meglitinides'     : ['repaglinide' , 'nateglinide'],
            'alpha_glucosidase': ['acarbose'    , 'miglitol'],
        }
        self.mapping_change = {'No': 0, 'Up': 1, 'Down': -1, 'Steady': 0}
        self.mapping_prescr = {'No': 0, 'Up': 1, 'Down': 1, 'Steady': 1}
        self.col_keep = []
        self.col_in = []
        self.handle_unknown = handle_unknown
        self.sparse_output = sparse_output
        self.is_fitted_ = False

    def process(self, X):
        X = X.copy()
        X[self.col_in] = X[self.col_in].fillna('No')

        df_prescr = pd.DataFrame(index=X.index)
        df_change = pd.DataFrame(index=X.index)
        df_final  = pd.DataFrame(index=X.index)

        # Map labels to numbers
        for col in self.col_in:
            if col in X.columns:
                df_prescr[col] = X[col].astype(str).map(self.mapping_prescr)
                df_change[col] = X[col].astype(str).map(self.mapping_change)

        # Extract labels from combo drugs and assign values to target drugs
        for combo, targets in self.col_combo.items():
            if combo in X.columns:
                for drug in targets:
                    if drug not in df_prescr.columns:
                        df_prescr[drug] = 0
                        df_change[drug] = 0
                    df_prescr[drug] = df_prescr[drug] + df_prescr[combo]
                    df_change[drug] = df_change[drug] + df_change[combo]
                df_prescr.drop(columns=combo, inplace=True)
                df_change.drop(columns=combo, inplace=True)

        # Process grouped drugs to calculate net prescribed drugs and net change in dosage
        for group, drugs in self.groups.items():
            drugs = [d for d in drugs if d in df_prescr.columns]
            if drugs:
                df_final[f'{group}__prescribed'] = df_prescr[drugs].sum(axis=1)
                df_final[f'{group}__change'] = df_change[drugs].sum(axis=1)

        # Process unique standalone drugs
        for drug in ['metformin', 'insulin', 'examide', 'citoglipton']:
            if drug in df_prescr.columns:
                df_final[f'{drug}__prescribed'] = df_prescr[drug]
                df_final[f'{drug}__change'] = df_change[drug]

        return df_final

    def fit(self, X, y=None):
        self.col_in = [c for c in self.columns if c in X.columns]
        df_final = self.process(X)
        self.col_keep = df_final.columns.tolist()
        if len(self.col_in) > 0:
            self.encoder_ = OneHotEncoder(
                handle_unknown=self.handle_unknown,
                sparse_output=self.sparse_output,
                # categories=[['No', 'Steady', 'Up', 'Down'] for _ in self.col_in]
            )
            self.encoder_.fit(X[self.col_in])

        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        df_final = self.process(X)

        columns = [c for c in self.col_in if c in X.columns]
        if len(columns) == 0:
            return pd.DataFrame(index=X.index)
        encoded = self.encoder_.transform(X[columns])
        oh_cols = self.encoder_.get_feature_names_out(self.col_in)
        df_oh = pd.DataFrame(encoded, index=X.index, columns=oh_cols)

        return pd.concat([df_final, df_oh], axis=1)

    def get_feature_names_out(self, input_features=None):
        oh_cols = (self.encoder_.get_feature_names_out(self.col_in) if hasattr(self, "encoder_") else [])
        return np.array(list(self.col_keep) + list(oh_cols))


class PipelineBuilder(object):

    CUSTOM = dict(
        medical_group        = MedicalSpecialtyGrouper,
        admission_type_group = AdmissionTypeIdGrouper,
        discharge_group      = DischargeDispositionIdGrouper,
        admission_src_group  = AdmissionSourceIdGrouper,
        icd9_group           = ICD9Grouper,
        high_no_drop         = HighNoDropper,
        prescription_group   = PrescriptionGrouper,
        drop                 = ColumnDropper,
    )
    ENCODERS = dict(
        k_bins  = KBinsDiscretizer,
        onehot  = OneHotEncoder,
        ordinal = OrdinalEncoder,
        target  = TargetEncoder,
    )
    IMPUTERS = dict(
        iterative = IterativeImputer,
        simple    = SimpleImputer,
    )
    SCALERS = dict(
        max_abs  = MaxAbsScaler,
        min_max  = MinMaxScaler,
        norm     = Normalizer,
        power    = PowerTransformer,
        quantile = QuantileTransformer,
        robust   = RobustScaler,
        standard = StandardScaler,
    )
    REGISTRIES = dict(
        imputer = IMPUTERS,
        scaler  = SCALERS,
        encoder = ENCODERS,
        custom  = CUSTOM,
    )

    def __init__(self, pipeline_config, column_config):

        pipeline_config = deepcopy(pipeline_config)

        for v in pipeline_config.values():
            v.update({'columns': []})

        for col, s in column_config.items():
            pipeline_config[s]['columns'].append(col)

        self.pipeline_config = pipeline_config
        self.column_config = column_config

    def build(self, column_transformer_param=None):

        column_transformer_param = column_transformer_param or dict()

        ls_transformers = []

        for k, v in self.pipeline_config.items():
            cols = v['columns']
            steps = v['steps']
            trans = Pipeline(steps=[
                (step['key'], self.REGISTRIES.get(step['registry']).get(step['key'])(**step.get('param', {})))
                for step in steps]
            )
            ls_transformers.append((k, trans, cols))

        return ColumnTransformer(ls_transformers, **column_transformer_param).set_output(transform='pandas')


def load_data(type='kf'):
    '''
    Load and preprocess data before pipeline.
    '''
    col_int = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
     ]

    data = dict()

    if type == 'kf':
        path = PATH_DATA_KF
    elif type == 'shuffle':
        path = PATH_DATA_SHUFFLE
    elif type == 'stratify':
        path = PATH_DATA_STRATIFY

    if type == 'kf':
        data['X_train_plus_val']  = pd.read_csv(path / 'X_train_plus_val.csv')
        data['X_train_mini']      = pd.read_csv(path / 'X_train_mini.csv')
        data['X_val']             = pd.read_csv(path / 'X_val.csv')
        data['X_test']            = pd.read_csv(path / 'X_test.csv')
        data['y_train_plus_val']  = pd.read_csv(path / 'y_train_plus_val.csv').astype(int)
        data['y_train_mini']      = pd.read_csv(path / 'y_train_mini.csv').astype(int)
        data['y_val']             = pd.read_csv(path / 'y_val.csv').astype(int)
        data['y_test']            = pd.read_csv(path / 'y_test.csv').astype(int)

    if type in ['shuffle', 'stratify']:
        data['X_train_for_cv']  = pd.read_csv(path / 'X_train_for_cv.csv')
        data['X_train_mini']    = pd.read_csv(path / 'X_train_mini.csv')
        data['X_val']           = pd.read_csv(path / 'X_val.csv')
        data['X_test']          = pd.read_csv(path / 'X_test.csv')
        data['y_train_for_cv']  = pd.read_csv(path / 'y_train_for_cv.csv').astype(int)
        data['y_train_mini']    = pd.read_csv(path / 'y_train_mini.csv').astype(int)
        data['y_val']           = pd.read_csv(path / 'y_val.csv').astype(int)
        data['y_test']          = pd.read_csv(path / 'y_test.csv').astype(int)

    for k, v in data.items():
        if k.startswith('X_'):
            data[k] = v.astype(str).replace('?', np.nan)
            data[k][col_int] = v[col_int].astype(int)

    return data


if __name__ == '__main__':

    # Configure column pipeline group
    COLUMN_CONFIG = {
        'encounter_id'            : 'pipeline_drop',
        # 'patient_nbr'             : 'pipeline_drop',          # Already dropped in split data
        'race'                    : 'pipeline_onehot',
        'gender'                  : 'pipeline_onehot',
        'age'                     : 'pipeline_onehot',          # or ordinal_encoding for NN
        'weight'                  : 'pipeline_drop',
        'admission_type_id'       : 'pipeline_admission_type',  # ro grouping + ordinal_encoding
        'discharge_disposition_id': 'pipeline_discharge',
        'admission_source_id'     : 'pipeline_admission_src',
        'payer_code'              : 'pipeline_drop',            # or 'impute_unknown'; grouping + onehot_encoding,
        'medical_specialty'       : 'pipeline_medical',         # or (grouping +) embedding (for NN)],
        'diag_1'                  : 'pipeline_icd9',            # or (grouping +) embedding (for NN)],
        'diag_2'                  : 'pipeline_icd9',            # or (grouping +) embedding (for NN)],
        'diag_3'                  : 'pipeline_icd9',            # or (grouping +) embedding (for NN)],
        'max_glu_serum'           : 'pipeline_onehot',          # or ordinal_encoding for NN
        'A1Cresult'               : 'pipeline_onehot',          # or ordinal_encoding for NN
        'metformin'               : 'pipeline_prescript',
        'repaglinide'             : 'pipeline_prescript',
        'nateglinide'             : 'pipeline_prescript',
        'chlorpropamide'          : 'pipeline_prescript',
        'glimepiride'             : 'pipeline_prescript',
        'acetohexamide'           : 'pipeline_prescript',
        'glipizide'               : 'pipeline_prescript',
        'glyburide'               : 'pipeline_prescript',
        'tolbutamide'             : 'pipeline_prescript',
        'pioglitazone'            : 'pipeline_prescript',
        'rosiglitazone'           : 'pipeline_prescript',
        'acarbose'                : 'pipeline_prescript',
        'miglitol'                : 'pipeline_prescript',
        'troglitazone'            : 'pipeline_prescript',
        'tolazamide'              : 'pipeline_prescript',
        'examide'                 : 'pipeline_prescript',
        'citoglipton'             : 'pipeline_prescript',
        'insulin'                 : 'pipeline_prescript',
        'glyburide-metformin'     : 'pipeline_prescript',
        'glipizide-metformin'     : 'pipeline_prescript',
        'glimepiride-pioglitazone': 'pipeline_prescript',
        'metformin-rosiglitazone' : 'pipeline_prescript',
        'metformin-pioglitazone'  : 'pipeline_prescript',
        'change'                  : 'pipeline_drop',
        'diabetesMed'             : 'pipeline_drop',
        'time_in_hospital'        : 'pipeline_standard',
        'num_lab_procedures'      : 'pipeline_standard',
        'num_procedures'          : 'pipeline_standard',
        'num_medications'         : 'pipeline_standard',
        'number_outpatient'       : 'pipeline_standard',
        'number_emergency'        : 'pipeline_standard',
        'number_inpatient'        : 'pipeline_standard',
        'number_diagnoses'        : 'pipeline_standard',
    }

    PIPELINE_CONFIG = {
        'pipeline_drop': {
            'steps': [
                {'registry': 'custom', 'key': 'drop', 'param': {}},
            ]
        },
        'pipeline_onehot': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_admission_type': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'custom', 'key': 'admission_type_group', 'param': {}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_discharge': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'custom', 'key': 'discharge_group', 'param': {}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_admission_src': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'custom', 'key': 'admission_src_group', 'param': {}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_medical': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'custom', 'key': 'medical_group', 'param': {}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_icd9': {
            'steps': [
                {'registry': 'imputer', 'key': 'simple', 'param': {'strategy': 'constant', 'fill_value': LABEL_UNKNOWN}},
                {'registry': 'custom', 'key': 'icd9_group', 'param': {}},
                {'registry': 'encoder', 'key': 'onehot', 'param': {'handle_unknown': 'ignore', 'sparse_output': False}},
            ]
        },
        'pipeline_prescript': {
            'steps': [
                {'registry': 'custom', 'key': 'high_no_drop', 'param': {'th': 1.}},
                {'registry': 'custom', 'key': 'prescription_group', 'param': {}},
            ]
        },
        'pipeline_standard': {
            'steps': [
                {'registry': 'scaler', 'key': 'standard', 'param': {}},
            ]
        },
    }

    # Load data
    data = load_data("kf")
    df = data['X_train_mini']

    pipeline = PipelineBuilder(PIPELINE_CONFIG, COLUMN_CONFIG).build()
    pipeline.fit(df)
    df_new = pipeline.transform(df)
