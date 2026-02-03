from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(df):
    categorical_cols = df.select_dtypes(include="object").columns
    numerical_cols = df.select_dtypes(exclude="object").columns.drop("Churn")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return preprocessor
