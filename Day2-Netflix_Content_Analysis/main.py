import sys
import os

sys.path.append(os.path.abspath("src"))



from src.data_loading import load_data
from src.data_cleaning import clean_data
from src.eda_visualization import plot_content_type, plot_genres
from src.feature_engineering import add_content_age
from src.wordcloud_analysis import generate_wordcloud

DATA_PATH = "/Users/vishalpande/Downloads/netflix_titles.csv"

df = load_data(DATA_PATH)
df = clean_data(df)

plot_content_type(df)
plot_genres(df)

df_age = add_content_age(df)
generate_wordcloud(df)
