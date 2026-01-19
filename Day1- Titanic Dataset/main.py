import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")


df = pd.read_csv(
    "/home/vkpande/21_days_21_projects/GenAI-Machine-Learning-Deep-Learning/Titanic-Dataset.csv"
)

# Normalize column names
df.columns = df.columns.str.lower()


print("Dataset Shape:", df.shape)
df.info()
print(df.columns)
print(df.describe(include="object"))

print(df.isnull().sum())

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()


# Age
if "age" in df.columns:
    df["age"] = df["age"].fillna(df["age"].median())

# Embarked 
if "embarked" in df.columns:
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Embark town
if "embark_town" in df.columns:
    df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# Drop high-missing columns safely
for col in ["deck", "cabin"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print("\nAfter Cleaning:\n", df.isnull().sum())

plt.figure(figsize=(6, 4))
sns.countplot(x="survived", data=df)
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Univariate Analysis

plt.figure(figsize=(8, 4))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="sex", data=df)
plt.title("Gender Distribution")
plt.tight_layout()
plt.show()

# Passenger class (Kaggle = pclass, Seaborn = class)
pclass_col = "pclass" if "pclass" in df.columns else "class"

plt.figure(figsize=(6, 4))
sns.countplot(x=pclass_col, data=df)
plt.title("Passenger Class Distribution")
plt.tight_layout()
plt.show()

# Bivariate Analysis


plt.figure(figsize=(6, 4))
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survival by Gender")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=pclass_col, hue="survived", data=df)
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="survived", y="age", data=df)
plt.title("Age vs Survival")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="survived", y="fare", data=df)
plt.title("Fare vs Survival")
plt.tight_layout()
plt.show()


# Multivariate Analysis

sns.catplot(
    x=pclass_col,
    hue="survived",
    col="sex",
    data=df,
    kind="count",
    height=4
)
plt.show()


# Feature Engineering

df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = np.where(df["family_size"] == 1, 1, 0)

plt.figure(figsize=(6, 4))
sns.boxplot(x="survived", y="family_size", data=df)
plt.title("Family Size vs Survival")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="is_alone", hue="survived", data=df)
plt.title("Survival vs Is Alone")
plt.tight_layout()
plt.show()


# Encoding for Correlation


df_encoded = df.copy()

if "sex" in df_encoded.columns:
    df_encoded["sex"] = df_encoded["sex"].map({"male": 0, "female": 1})

if pclass_col in df_encoded.columns:
    df_encoded[pclass_col] = df_encoded[pclass_col].astype("category").cat.codes

if "embarked" in df_encoded.columns:
    df_encoded["embarked"] = df_encoded["embarked"].astype("category").cat.codes

plt.figure(figsize=(12, 8))
sns.heatmap(
    df_encoded.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# Y-Profiling Report


profile = ProfileReport(
    df,
    title="Titanic Dataset - Y Profiling Report",
    explorative=True
)

profile.to_file("titanic_y_profiling_report.html")

print(" Y-Profiling report generated: titanic_y_profiling_report.html")
