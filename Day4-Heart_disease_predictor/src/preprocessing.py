import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Binary target
    df['heart_disease'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    # Drop unused columns
    df = df.drop(columns=['id', 'dataset', 'num'])

    # Fill missing values
    for col in df.select_dtypes(include='float64').columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # One-hot encoding (manual)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('heart_disease', axis=1)
    y = df['heart_disease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
