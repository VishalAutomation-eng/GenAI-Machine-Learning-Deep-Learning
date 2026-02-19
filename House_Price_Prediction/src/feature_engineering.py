import pandas as pd

def add_features(all_data):
    all_data['TotalSF'] = (
        all_data['TotalBsmtSF'] +
        all_data['1stFlrSF'] +
        all_data['2ndFlrSF']
    )

    all_data['TotalBath'] = (
        all_data['FullBath'] +
        0.5 * all_data['HalfBath'] +
        all_data['BsmtFullBath'] +
        0.5 * all_data['BsmtHalfBath']
    )

    all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']

    return all_data


def encode_features(all_data):
    categorical_cols = all_data.select_dtypes(include='object').columns
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    return all_data
