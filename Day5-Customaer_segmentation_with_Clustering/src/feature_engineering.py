def add_spending_income_ratio(df):
    """
    Feature Engineering:
    Spending to Income Ratio
    """
    df["Spending_Income_Ratio"] = (
        df["Spending Score (1-100)"] / df["Annual Income (k$)"]
    )
    return df
