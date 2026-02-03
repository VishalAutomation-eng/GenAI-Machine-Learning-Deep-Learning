import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_clustering(df):
    features = df[["Annual Income (k$)", "Spending_Income_Ratio"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, 11), wcss, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.savefig("outputs/elbow_method.png")
    plt.show()

    # Final Model
    kmeans = KMeans(n_clusters=5, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(8,5))
    plt.scatter(
        df["Annual Income (k$)"],
        df["Spending_Income_Ratio"],
        c=df["Cluster"],
        cmap="viridis"
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending-Income Ratio")
    plt.title("Customer Segments")
    plt.savefig("outputs/customer_clusters.png")
    plt.show()

    return df
