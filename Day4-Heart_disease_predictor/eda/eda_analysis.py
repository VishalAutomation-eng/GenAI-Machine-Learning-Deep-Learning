import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("\nDataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    # Create binary target
    df['heart_disease'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

    # Target distribution
    sns.countplot(x='heart_disease', data=df)
    plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
    plt.show()

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(12,8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

    # Age vs heart disease
    sns.boxplot(x='heart_disease', y='age', data=df)
    plt.title("Age vs Heart Disease")
    plt.show()
