import os
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from icd9cms.icd9 import search


def get_embedder(model_name='FremyCompany/BioLORD-2023',
                 cache_folder=None,
                 device='cpu'
                 ):
    """
    Create an embedder.

    Args:
        model_name: Huggingface model name. Default to BioLORD-2023.
        cache_folder: Target cache folder to save the model
        device: Device to use

    Returns:
        An embedder
    """
    model = SentenceTransformer(model_name,
                                cache_folder=cache_folder,
                                device=device)
    return model


def icd9_to_desc(code, col):
    """
    Convert ICD-9 code to description for diagnosis features.

    Args:
        code: ICD-9 code in string
        col: Either `diag_1` or `diag_2` or `diag_3`

    Return:
         Diagnosis description at the current ICD-9 level in string.
    """
    tier_map = {
        'diag_1': 'Primary Diagnosis',
        'diag_2': 'Secondary Diagnosis 1',
        'diag_3': 'Secondary Diagnosis 2',
    }
    tier = tier_map[col]
    if code not in [np.nan, '?', '', None]:
        code = str(code)
        if code.isnumeric():
            code = code.zfill(3)
        desc = search(code)
        out = f'{tier.title()} - [ICD-9 {code}]: {desc.short_desc}'
        if desc.long_desc:
            out += f' ({desc.long_desc})'
        return out
    else:
        return np.nan


def convert_icd9(df):
    """
    Convert ICD-9 code to description for diagnosis features.

    Args:
        df: DataFrame with ICD-9 code columns diag_1, diag_2, diag_3

    Returns:
        DataFrame with ICD-9 description with column names diag_1_desc, diag_2_desc, diag_3_desc.
    """
    cols_diag = ['diag_1', 'diag_2', 'diag_3']
    df_out = pd.DataFrame(index=df.index)
    for c in cols_diag:
        df_out[c + '_desc'] = df[c].apply(lambda x: icd9_to_desc(x, col=c))
    return df_out


def render_atlas(path, param):
    """
    Render embedding-atlas app.

    Args:
        path: target file path
        param: dictionary of command line arguments
    """
    arg = ''
    for k, v in param.items():
        arg += f' --{k} {v}'
    cmd = f'embedding-atlas {path}' + arg
    os.system(cmd)
