import os
import json
import pandas as pd
import numpy as np
import joblib
import shap
import pickle
import matplotlib.pyplot as plt

if not hasattr(np, "in1d"):
    np.in1d = np.isin

import dalex as dx
from pathlib import Path
from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder, OrdinalEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, fbeta_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, average_precision_score
from src.utils.data_fetch import DataLoader
from src.stats import compare_adjacent_models
from src.utils.model_eval.evaluation import evaluate_predictions
from src.visualisations.viz_model import plot_confusion_matrix


# Settings -------------------------------------------------------------------------------------------

SEED = 1234
# PATH_XGB = Path('../')
# PATH_TE_DATA = Path('../../../data/extra/target_encoded')
# PATH_LABEL_MAP = Path('../../../data/final_processed/base_kf/label/label_map.json')
# PATH_XGB_ANALYSIS = Path('../../../data/extra/analysis')
PATH_ROOT = Path(os.getcwd())
PATH_XGB = PATH_ROOT / r'notebook\gradient_boosting'
PATH_TE_DATA = PATH_ROOT / r'data\extra\target_encoded'
PATH_LABEL_MAP = PATH_ROOT / r'data\final_processed\base_kf\label\label_map.json'
PATH_XGB_ANALYSIS = PATH_ROOT / r'data\extra\analysis'
DICT_PREFIX = {
    "m0": "xgboost_onehot",
    "m1": "xgboost_num",
    "m2": "xgboost_num_dem",
    "m3": "xgboost_num_dem_clin",
    "m4": "xgboost_num_dem_clin_diag",
    "m5": "xgboost_num_dem_clin_diag_druggrp",
    "m6": "xgboost_label",
    "m7": "xgboost_full_te_clin",
    "m8": "xgboost_full_te_clin_diag",
    "m9": "xgboost_full_te_clin_diag_other",
    # 'm10': 'xgboost_final',
}
TUNE_METRIC = 'prauc' # recall/auc/f1/f2/prauc
DICT_PATH_PARAM        = {k: v + '_' + f'params_{TUNE_METRIC}.json' for k, v in DICT_PREFIX.items()}
DICT_PATH_LOG_OUTPUT   = {k: v + '_' + f'log_{TUNE_METRIC}.csv' for k    , v in DICT_PREFIX.items()}
DICT_PATH_PROB_OUTPUT  = {k: v + '_' + f'prob_{TUNE_METRIC}.json' for k  , v in DICT_PREFIX.items()}
DICT_PATH_MODEL_OUTPUT = {k: v + '_' + f'model_{TUNE_METRIC}.json' for k , v in DICT_PREFIX.items()}
DICT_FEATURE           = dict()
DICT_FEATURE_GROUP     = dict()
fixed_params = {
    'objective'         : 'binary:logistic',
    'eval_metric'       : 'logloss',
    'tree_method'       : 'hist',
    'device'            : 'cpu',
    'enable_categorical': True,
    'random_state'      : SEED,
    'n_jobs'            : -1,
}

# Load Outputs -------------------------------------------------------------------------------------------

# Tuning log
DICT_LOG_OUTPUT = {k: pd.read_csv(PATH_XGB / v) for k, v in DICT_PATH_LOG_OUTPUT.items()}

# Tuned model
DICT_MODEL_OUTPUT = {k: joblib.load(PATH_XGB / v) for k, v in DICT_PATH_MODEL_OUTPUT.items()}

# Tuned probability threshold
DICT_PROB_OUTPUT = dict()
for k, v in DICT_PATH_PROB_OUTPUT.items():
    with open(PATH_XGB / v, 'r') as f:
        DICT_PROB_OUTPUT[k] = json.load(f)['best_th']

# Tuned parameters
DICT_PARAM = dict()
for k, v in DICT_PATH_PARAM.items():
    with open(PATH_XGB / v, 'r') as f:
        best_params = json.load(f)
    final_params = deepcopy(fixed_params)
    final_params.update(best_params)
    final_params.update({"device": "cpu"})
    DICT_PARAM[k] = final_params


# Load and Preprocess Data -------------------------------------------------------------------------------------------

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = DataLoader().fetch_data(version='final_raw')
X_train_label, X_val_label, X_test_label, _, _, _ = DataLoader().fetch_data(version='final_label')
X_train_oh, X_val_oh, X_test_oh, _, _, _ = DataLoader().fetch_data(version='final_onehot')
X_train_interim, X_val_interim, X_test_interim, _, _, _ = DataLoader().fetch_data(version='interim')
df_raw = pd.concat([X_train_interim, X_val_interim, X_test_interim], axis=0, ignore_index=True)

# Update one-hot data columns
cols_update = [c.replace('[', '(').replace(')', ')').replace('<', 'lt').replace('>', 'gt') for c in X_train_oh.columns]
X_train_oh.columns = cols_update
X_val_oh.columns = cols_update
X_test_oh.columns = cols_update

# Define columns
id_cols = [
    'pipeline_raw__encounter_id'
]
num_cols = [
    'pipeline_standard__time_in_hospital',
    'pipeline_standard__num_lab_procedures',
    'pipeline_standard__num_procedures',
    'pipeline_standard__num_medications',
    'pipeline_standard__number_outpatient',
    'pipeline_standard__number_emergency',
    'pipeline_standard__number_inpatient',
    'pipeline_standard__number_diagnoses'
]
dem_cols = [
    'pipeline_raw__race',
    'pipeline_raw__gender',
    'pipeline_raw__age',
]
clin_cols = [
    'pipeline_raw__max_glu_serum',
    'pipeline_raw__A1Cresult',
    'pipeline_admission_type_raw__admission_type_id',
    'pipeline_discharge_raw__discharge_disposition_id',
    'pipeline_admission_src_raw__admission_source_id',
    'pipeline_medical_raw__medical_specialty',
]
diag_cols = [
    'pipeline_icd9_raw__diag_1',
    'pipeline_icd9_raw__diag_2',
    'pipeline_icd9_raw__diag_3',
]
druggrp_cols = [
    'pipeline_prescript_raw__sulfonylureas__prescribed',
    'pipeline_prescript_raw__sulfonylureas__change',
    'pipeline_prescript_raw__tzds__prescribed',
    'pipeline_prescript_raw__tzds__change',
    'pipeline_prescript_raw__meglitinides__prescribed',
    'pipeline_prescript_raw__meglitinides__change',
    'pipeline_prescript_raw__alpha_glucosidase__prescribed',
    'pipeline_prescript_raw__alpha_glucosidase__change',
    'pipeline_prescript_raw__metformin__prescribed',
    'pipeline_prescript_raw__metformin__change',
    'pipeline_prescript_raw__insulin__prescribed',
    'pipeline_prescript_raw__insulin__change',
]
drug_cols = [
    'pipeline_prescript_raw__metformin',
    'pipeline_prescript_raw__repaglinide',
    'pipeline_prescript_raw__nateglinide',
    'pipeline_prescript_raw__chlorpropamide',
    'pipeline_prescript_raw__glimepiride',
    'pipeline_prescript_raw__acetohexamide',
    'pipeline_prescript_raw__glipizide',
    'pipeline_prescript_raw__glyburide',
    'pipeline_prescript_raw__tolbutamide',
    'pipeline_prescript_raw__pioglitazone',
    'pipeline_prescript_raw__rosiglitazone',
    'pipeline_prescript_raw__acarbose',
    'pipeline_prescript_raw__miglitol',
    'pipeline_prescript_raw__troglitazone',
    'pipeline_prescript_raw__tolazamide',
    'pipeline_prescript_raw__insulin',
    'pipeline_prescript_raw__glyburide-metformin',
    'pipeline_prescript_raw__glipizide-metformin',
    'pipeline_prescript_raw__glimepiride-pioglitazone',
    'pipeline_prescript_raw__metformin-rosiglitazone',
    'pipeline_prescript_raw__metformin-pioglitazone',
]
diag_te_cols  = [
    'diag_1_te',
    'diag_2_te',
    'diag_3_te',
]
clin_te_cols = [
    'medical_specialty_te',
    'admission_source_id_te',
    'admission_type_id_te',
    'discharge_disposition_id_te',
]
other_cols = [
    'weight',
    'payer_code'
]
num_cols_raw = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient',
    'number_emergency',
    'number_inpatient',
    'number_diagnoses'
]

# Drop columns
X_train    = X_train.drop(id_cols + num_cols_raw, axis = 1)
X_val      = X_val.drop(id_cols + num_cols_raw  , axis = 1)
X_test     = X_test.drop(id_cols + num_cols_raw , axis = 1)
X_train_oh = X_train_oh.drop(id_cols            , axis = 1)
X_val_oh   = X_val_oh.drop(id_cols              , axis = 1)
X_test_oh  = X_test_oh.drop(id_cols             , axis = 1)

# Use standardize numerical columns in raw
X_train[num_cols] = X_train_label[num_cols]
X_val[num_cols] = X_val_label[num_cols]
X_test[num_cols] = X_test_label[num_cols]

# Add target encoded columns
X_train_te = pd.read_csv(PATH_TE_DATA / "X_train_te.csv")
X_val_te   = pd.read_csv(PATH_TE_DATA / "X_val_te.csv")
X_test_te  = pd.read_csv(PATH_TE_DATA / "X_test_te.csv")
X_train[clin_te_cols + diag_te_cols] = X_train_te[clin_te_cols + diag_te_cols]
X_val[clin_te_cols + diag_te_cols]   = X_val_te[clin_te_cols + diag_te_cols]
X_test[clin_te_cols + diag_te_cols]  = X_test_te[clin_te_cols + diag_te_cols]

# Add other columns (weight, payer_code)
df_raw['payer_code'] = df_raw['payer_code'].replace('?'   , np.nan).fillna('Unknown').astype('category')
df_raw['weight']     = df_raw['weight'].replace('?'       , np.nan).fillna('Unknown').astype('category')
all_payer_code       = df_raw['payer_code'].cat.categories
all_weight           = df_raw['weight'].cat.categories

X_train['payer_code'] = pd.Categorical(X_train_interim['payer_code'].replace('?', np.nan).fillna('Unknown'), categories = all_payer_code)
X_val['payer_code']   = pd.Categorical(X_val_interim['payer_code'].replace('?'  , np.nan).fillna('Unknown'), categories = all_payer_code)
X_test['payer_code']  = pd.Categorical(X_test_interim['payer_code'].replace('?' , np.nan).fillna('Unknown'), categories = all_payer_code)
X_train['weight']     = pd.Categorical(X_train_interim['weight'].replace('?'    , np.nan).fillna('Unknown'), categories = all_weight)
X_val['weight']       = pd.Categorical(X_val_interim['weight'].replace('?'      , np.nan).fillna('Unknown'), categories = all_weight)
X_test['weight']      = pd.Categorical(X_test_interim['weight'].replace('?'     , np.nan).fillna('Unknown'), categories = all_weight)

# Convert categorical columns type
cat_cols = [c for c in X_train.columns if c not in num_cols + druggrp_cols + clin_te_cols + diag_te_cols]
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_val[cat_cols]   = X_val[cat_cols].astype('category')
X_test[cat_cols]  = X_test[cat_cols].astype('category')

# Populate column dictionary for each model
DICT_FEATURE['m0'] = X_train_oh.columns.tolist()
DICT_FEATURE['m1'] = num_cols
DICT_FEATURE['m2'] = num_cols + dem_cols
DICT_FEATURE['m3'] = num_cols + dem_cols + clin_cols
DICT_FEATURE['m4'] = num_cols + dem_cols + clin_cols + diag_cols
DICT_FEATURE['m5'] = num_cols + dem_cols + clin_cols + diag_cols + druggrp_cols
DICT_FEATURE['m6'] = num_cols + dem_cols + clin_cols + diag_cols + druggrp_cols + drug_cols
DICT_FEATURE['m7'] = num_cols + dem_cols + clin_cols + diag_cols + druggrp_cols + drug_cols + clin_te_cols
DICT_FEATURE['m8'] = num_cols + dem_cols + clin_cols + diag_cols + druggrp_cols + drug_cols + clin_te_cols + diag_te_cols
DICT_FEATURE['m9'] = num_cols + dem_cols + clin_cols + diag_cols + druggrp_cols + drug_cols + clin_te_cols + diag_te_cols + other_cols
# DICT_FEATURE['m10'] = ...

DICT_ADDED_FEATURES = dict()
DICT_ADDED_FEATURES['m0'] = 'One-Hot (Reference Only)'
DICT_ADDED_FEATURES['m1'] = 'Numerical (Raw, Baseline)'
DICT_ADDED_FEATURES['m2'] = 'Demographics (Raw)'
DICT_ADDED_FEATURES['m3'] = 'Clinical (Grouped)'
DICT_ADDED_FEATURES['m4'] = 'Diagnosis (Grouped)'
DICT_ADDED_FEATURES['m5'] = 'Prescription Statistics (Grouped)'
DICT_ADDED_FEATURES['m6'] = 'Prescription (Raw)'
DICT_ADDED_FEATURES['m7'] = 'Clinical (Raw, Target Encoded)'
DICT_ADDED_FEATURES['m8'] = 'Diagnosis (Raw, Target Encoded)'
DICT_ADDED_FEATURES['m9'] = 'Payer Code & Weight (Raw)'


RUN_PREDICTIVE_CONTRI = False
RUN_FEATURE_GROUP_IMP_CONCEPT = False
RUN_FEATURE_GROUP_IMP_REP = False
RUN_SHAP_IMPORTANCE = False

# 1. Predictive Contribution -------------------------------------------------------------------------------------------

if RUN_PREDICTIVE_CONTRI:

    ls_perf = []
    dict_prob = dict()

    # Get predictive performances using validation data
    for k, model in DICT_MODEL_OUTPUT.items():

        print(k)

        features = model.feature_names_in_
        if k == 'm0':
            y_prob_val = model.predict_proba(X_val_oh[features])[:, 1]
            y_pred_val = model.predict(X_val_oh[features])
        else:
            y_prob_val = model.predict_proba(X_val[features])[:, 1]
            y_pred_val = model.predict(X_val[features])

        dict_perf = dict(
            model        = k,
            new_features = DICT_ADDED_FEATURES[k],
            recall       = recall_score(y_val           , y_pred_val),
            precision    = precision_score(y_val        , y_pred_val),
            accuracy     = accuracy_score(y_val         , y_pred_val),
            roc_auc      = roc_auc_score(y_val          , y_prob_val),
            prauc        = average_precision_score(y_val, y_prob_val),
            f1           = f1_score(y_val               , y_pred_val),
            f2           = fbeta_score(y_val            , y_pred_val , beta = 2),
        )
        ls_perf.append(dict_perf)

        dict_prob[k] = y_prob_val

    df_perf = pd.DataFrame(ls_perf)
    df_perf['d_prauc'] = df_perf['prauc'].diff()
    df_perf.loc[df_perf['model'].isin(['m0', 'm1']), 'd_prauc'] = np.nan
    df_perf.set_index('model', inplace=True)
    df_perf.to_csv(PATH_XGB_ANALYSIS / 'ana1_perf_contri.csv')

    # Get PRAUC comparisons
    df_prauc = compare_adjacent_models(y_val.values, dict_prob, n_bootstrap=5000, random_state=42)
    df_prauc.set_index('comparison', inplace=True)
    df_prauc.to_csv(PATH_XGB_ANALYSIS / 'ana1_perf_contri_prauc.csv')

else:
    df_perf = pd.read_csv(PATH_XGB_ANALYSIS / 'ana1_perf_contri.csv', index_col=0)
    df_prauc = pd.read_csv(PATH_XGB_ANALYSIS / 'ana1_perf_contri_prauc.csv', index_col=0)


# 2. Concept (Feature Group) Importance -------------------------------------------------------------------------------------------

# Define prediction function
def predict_positive_prob(model, data):
    data = data.loc[:, model.feature_names_in_]
    return model.predict_proba(data)[:, 1]

# Define metric
def negative_prauc(obs, pred):
    return -average_precision_score(obs, pred)

# Define feature groups
group_visit = [
    "pipeline_standard__number_outpatient",
    "pipeline_standard__number_emergency",
    "pipeline_standard__number_inpatient",
]
group_hosp_util = [
    'pipeline_standard__time_in_hospital',
    'pipeline_standard__num_lab_procedures',
    'pipeline_standard__num_procedures',
    'pipeline_standard__num_medications',
    'pipeline_standard__number_diagnoses'
]
group_clin = [
    'pipeline_raw__max_glu_serum',
    'pipeline_raw__A1Cresult',
    'pipeline_admission_type_raw__admission_type_id',
    'pipeline_discharge_raw__discharge_disposition_id',
    'pipeline_admission_src_raw__admission_source_id',
    'pipeline_medical_raw__medical_specialty',
    'medical_specialty_te',
    'admission_source_id_te',
    'admission_type_id_te',
    'discharge_disposition_id_te',
]
group_diag = [
    "pipeline_icd9_raw__diag_1",
    "pipeline_icd9_raw__diag_2",
    "pipeline_icd9_raw__diag_3",
    "diag_1_te",
    "diag_2_te",
    "diag_3_te",
]
group_med = [
    'pipeline_prescript_raw__sulfonylureas__prescribed',
    'pipeline_prescript_raw__sulfonylureas__change',
    'pipeline_prescript_raw__tzds__prescribed',
    'pipeline_prescript_raw__tzds__change',
    'pipeline_prescript_raw__meglitinides__prescribed',
    'pipeline_prescript_raw__meglitinides__change',
    'pipeline_prescript_raw__alpha_glucosidase__prescribed',
    'pipeline_prescript_raw__alpha_glucosidase__change',
    'pipeline_prescript_raw__metformin__prescribed',
    'pipeline_prescript_raw__metformin__change',
    'pipeline_prescript_raw__insulin__prescribed',
    'pipeline_prescript_raw__insulin__change',
    'pipeline_prescript_raw__metformin',
    'pipeline_prescript_raw__repaglinide',
    'pipeline_prescript_raw__nateglinide',
    'pipeline_prescript_raw__chlorpropamide',
    'pipeline_prescript_raw__glimepiride',
    'pipeline_prescript_raw__acetohexamide',
    'pipeline_prescript_raw__glipizide',
    'pipeline_prescript_raw__glyburide',
    'pipeline_prescript_raw__tolbutamide',
    'pipeline_prescript_raw__pioglitazone',
    'pipeline_prescript_raw__rosiglitazone',
    'pipeline_prescript_raw__acarbose',
    'pipeline_prescript_raw__miglitol',
    'pipeline_prescript_raw__troglitazone',
    'pipeline_prescript_raw__tolazamide',
    'pipeline_prescript_raw__insulin',
    'pipeline_prescript_raw__glyburide-metformin',
    'pipeline_prescript_raw__glipizide-metformin',
    'pipeline_prescript_raw__glimepiride-pioglitazone',
    'pipeline_prescript_raw__metformin-rosiglitazone',
    'pipeline_prescript_raw__metformin-pioglitazone',
]

feature_groups_concept = {
    'visit': group_visit,
    'demographics': dem_cols,
    'hospital_util': group_hosp_util,
    'clinical': group_clin,
    'diagnosis': group_diag,
    'medication': group_med,
    'other': other_cols,
}

# Run feature concept importance
if RUN_FEATURE_GROUP_IMP_CONCEPT:

    # Get last model and data
    k = "m9"
    model = DICT_MODEL_OUTPUT[k]
    x = X_val[model.feature_names_in_]

    explainer_concept = dx.Explainer(model, x, y_val, label=k, predict_function=predict_positive_prob, verbose=True)
    importance_concept = explainer_concept.model_parts(variable_groups=feature_groups_concept,
                                                       loss_function=negative_prauc,
                                                       type='difference',
                                                       N=None,
                                                       B=100,        # num of permutation
                                                       processes=1,
                                                       random_state=42)

    with open(PATH_XGB_ANALYSIS / 'ana2_importance_concept.pkl', 'wb') as f:
        pickle.dump(importance_concept, f)

else:

    with open(PATH_XGB_ANALYSIS / 'ana2_importance_concept.pkl', 'rb') as f:
        importance_concept = pickle.load(f)

# Plot importance
importance_concept.plot(show=False).show(renderer='browser')


# 3. Representation Analysis -------------------------------------------------------------------------------------------

feature_groups_diag_sep = {
    'diagnosis_grouped': diag_cols,
    'diagnosis_te': diag_te_cols
}
feature_groups_diag_joint = {
    'diagnosis_joint': diag_cols + diag_te_cols
}
feature_groups_clin_sep = {
    'clinical_grouped': [
        "pipeline_admission_type_raw__admission_type_id",
        "pipeline_discharge_raw__discharge_disposition_id",
        "pipeline_admission_src_raw__admission_source_id",
        "pipeline_medical_raw__medical_specialty",
    ],
    'clinical_te': clin_te_cols,
}
feature_groups_clin_joint = {
    'clinical_joint': [
        "pipeline_admission_type_raw__admission_type_id",
        "pipeline_discharge_raw__discharge_disposition_id",
        "pipeline_admission_src_raw__admission_source_id",
        "pipeline_medical_raw__medical_specialty",
    ] + clin_te_cols
}

# Run feature concept importance
if RUN_FEATURE_GROUP_IMP_REP:

    # Get last model and data
    k = "m9"
    model = DICT_MODEL_OUTPUT[k]
    x = X_val[model.feature_names_in_]

    explainer_rep = dx.Explainer(model, x, y_val, label=k, predict_function=predict_positive_prob, verbose=True)

    print('Running Importance: Diagnosis (Separated)')
    importance_diag_sep = explainer_rep.model_parts(variable_groups=feature_groups_diag_sep,
                                                    loss_function=negative_prauc,
                                                    type='difference',
                                                    N=None,
                                                    B=100,
                                                    processes=1,
                                                    random_state=42)

    print('Running Importance: Diagnosis (Joint)')
    importance_diag_joint = explainer_rep.model_parts(variable_groups=feature_groups_diag_joint,
                                                      loss_function=negative_prauc,
                                                      type='difference',
                                                      N=None,
                                                      B=100,
                                                      processes=1,
                                                      random_state=42)

    print('Running Importance: Clinical (Separated)')
    importance_clin_sep = explainer_rep.model_parts(variable_groups=feature_groups_clin_sep,
                                                    loss_function=negative_prauc,
                                                    type='difference',
                                                    N=None,
                                                    B=100,
                                                    processes=1,
                                                    random_state=42)

    print('Running Importance: Clinical (Joint)')
    importance_clin_joint = explainer_rep.model_parts(variable_groups=feature_groups_clin_joint,
                                                      loss_function=negative_prauc,
                                                      type='difference',
                                                      N=None,
                                                      B=100,
                                                      processes=1,
                                                      random_state=42)


    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_diag_sep.pkl', 'wb') as f:
        pickle.dump(importance_diag_sep, f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_diag_joint.pkl', 'wb') as f:
        pickle.dump(importance_diag_joint, f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_clin_sep.pkl', 'wb') as f:
        pickle.dump(importance_clin_sep, f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_clin_joint.pkl', 'wb') as f:
        pickle.dump(importance_clin_joint, f)

else:

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_diag_sep.pkl', 'rb') as f:
        importance_diag_sep = pickle.load(f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_diag_joint.pkl', 'rb') as f:
        importance_diag_joint = pickle.load(f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_clin_sep.pkl', 'rb') as f:
        importance_clin_sep = pickle.load(f)

    with open(PATH_XGB_ANALYSIS / 'ana3_importance_rep_clin_joint.pkl', 'rb') as f:
        importance_clin_joint = pickle.load(f)

# Plot importance
importance_diag_sep.plot(show=False).show(renderer='browser')
importance_diag_joint.plot(show=False).show(renderer='browser')
importance_clin_sep.plot(show=False).show(renderer='browser')
importance_clin_joint.plot(show=False).show(renderer='browser')

# 4. Behaviour Analysis -------------------------------------------------------------------------------------------

def aggregate_shap_groups(shap_values,
                          feature_names,
                          groups
                          ):
    feature_index = {name: i for i, name in enumerate(feature_names)}
    output = {}

    for group_name, columns in groups.items():
        indices = [feature_index[column] for column in columns if column in feature_index]
        output[group_name] = shap_values.values[:, indices].sum(axis=1)

    return pd.DataFrame(output)

if RUN_SHAP_IMPORTANCE:

    # # Importance heatmap
    DICT_SHAP_EXPLAINER = {k: shap.TreeExplainer(v) for k, v in DICT_MODEL_OUTPUT.items() if k != 'm0'}
    DICT_SHAP_EXPLAINER_COLS = {k: v.feature_names_in_ for k, v in DICT_MODEL_OUTPUT.items() if k != 'm0'}
    DICT_SHAP_VALUES = {k: v(X_val[DICT_MODEL_OUTPUT[k].feature_names_in_]) for k, v in DICT_SHAP_EXPLAINER.items()}
    for k, v in DICT_SHAP_VALUES.items():
        v.feature_names = ['__'.join(c.split('__')[1:]) if len(c.split('__')) > 1 else c for c in v.feature_names]

    ls_shap = []
    for k, v in DICT_SHAP_VALUES.items():
        feature_imp = v.abs.mean(axis=0).values
        df_shap = pd.DataFrame({'feature': v.feature_names, k: feature_imp}).set_index('feature')
        ls_shap.append(df_shap)
    df_shap_all = pd.concat(ls_shap, axis=1).sort_values('m9', ascending=False)
    df_shap_all.to_csv(PATH_XGB_ANALYSIS / 'ana4_shap_feature_importance.csv')

    with open(PATH_XGB_ANALYSIS / "ana4_shap_values.pkl", "wb") as f:
        pickle.dump(DICT_SHAP_VALUES, f)

else:

    with open(PATH_XGB_ANALYSIS / "ana4_shap_values.pkl", "rb") as f:
        DICT_SHAP_VALUES = pickle.load(f)

    df_shap_all = pd.read_csv(PATH_XGB_ANALYSIS / 'ana4_shap_feature_importance.csv', index_col=0)


# Create beeswarm plot
shap_values = DICT_SHAP_VALUES['m9']
shap.plots.beeswarm(
    shap_values,
    max_display=34,   # selected based on >= 0.01
    group_remaining_features=False,
)

# Define feature groups
feature_groups_concept_adj = deepcopy(feature_groups_concept)
for k, v in feature_groups_concept_adj.items():
    feature_groups_concept_adj[k] = ['__'.join(c.split('__')[1:]) if len(c.split('__')) > 1 else c for c in v]

# Get feature concept shap
df_shap_concept = aggregate_shap_groups(shap_values, shap_values.feature_names, feature_groups_concept_adj)
# df_shap_concept.to_csv(PATH_XGB_ANALYSIS / 'ana4_shap_concept.csv', index=False)

# Get mean feature concept shap
df_shap_concept_mean = df_shap_concept.mean().sort_values(ascending=False).reset_index()
df_shap_concept_mean.columns = ['concept', 'mean_shap']
# df_shap_concept_mean.to_csv(PATH_XGB_ANALYSIS / 'ana4_shap_concept_mean.csv', index=False)

# Get mean abs feature concept shap
df_shap_concept_mean_abs = df_shap_concept.abs().mean().sort_values(ascending=False).reset_index()
df_shap_concept_mean_abs.columns = ['concept', 'mean_shap']
# df_shap_concept_mean_abs.to_csv(PATH_XGB_ANALYSIS / 'ana4_shap_concept_mean_abs.csv', index=False)

# TODO: te

# Dependency plot
if False:
    shap.plots.scatter(shap_values[:, 'diag_1'])
    shap.plots.scatter(shap_values[:, 'diag_2'])
    shap.plots.scatter(shap_values[:, 'diag_3'])
    shap.plots.scatter(shap_values[:, 'payer_code'])
    shap.plots.scatter(shap_values[:, 'medical_specialty'])
    shap.plots.scatter(shap_values[:, 'age'])
    shap.plots.scatter(shap_values[:, 'weight'])
    shap.plots.scatter(shap_values[:, 'race'])
    shap.plots.scatter(shap_values[:, 'gender'])


# Subgroup Prediction Analysis -------------------------------------------------------------------------------------------
# Error Analysis -------------------------------------------------------------------------------------------
# Interaction Analysis -------------------------------------------------------------------------------------------
# Subgroup Fairness Analysis -------------------------------------------------------------------------------------------
# Individual Explanation -------------------------------------------------------------------------------------------

