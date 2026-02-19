import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def plot_content_type(df):
    type_counts = df['type'].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%')
    plt.title('Movies vs TV Shows')
    plt.show()


def plot_genres(df):
    genres = df.assign(genre=df['listed_in'].str.split(', ')).explode('genre')
    top_genres = genres['genre'].value_counts().head(15)

    plt.figure(figsize=(10,6))
    sns.barplot(y=top_genres.index, x=top_genres.values)
    plt.title('Top 15 Genres on Netflix')
    plt.show()
