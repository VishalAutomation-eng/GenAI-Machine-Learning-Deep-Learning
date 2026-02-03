import numpy as np
import pandas as pd

def handle_missing_values(all_data):
    numerical_zero_cols = [
        'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars',
        'GarageArea', 'BsmtFullBath', 'BsmtHalfBath'
    ]

    for col in numerical_zero_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)

    if 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby(
            'Neighborhood'
        )['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    categorical_none_cols = [
        'Alley', 'Fence', 'MiscFeature', 'PoolQC',
        'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'MasVnrType'
    ]

    for col in categorical_none_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna('None')

    if 'GarageYrBlt' in all_data.columns:
        all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

    return all_data
