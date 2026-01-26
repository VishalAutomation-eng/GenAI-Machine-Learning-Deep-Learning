import matplotlib.pyplot as plt
import seaborn as sns

def gender_vs_spending(df):
    print("\nSummary Statistics by Gender:")
    print(df.groupby("Gender")["Spending Score (1-100)"].describe())

    plt.figure(figsize=(8,5))
    sns.boxplot(x="Gender", y="Spending Score (1-100)", data=df)
    plt.title("Gender vs Spending Score")
    plt.savefig("outputs/gender_vs_spending.png")
    plt.show()
