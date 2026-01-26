from src.data_loader import load_data
from src.gender_spending_analysis import gender_vs_spending
from src.feature_engineering import add_spending_income_ratio
from src.clustering import perform_clustering

if __name__ == "__main__":
    df = load_data("/Users/vishalpande/Downloads/Mall_Customers.csv")

    # Criteria 1
    gender_vs_spending(df)

    # Criteria 2
    df = add_spending_income_ratio(df)
    df = perform_clustering(df)

    print("\nFinal Data with Clusters:")
    print(df.head())
