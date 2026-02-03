import pandas as pd

def clean_data(df):
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')

    mode_country = df['country'].mode()[0]
    df['country'] = df['country'].fillna(mode_country)

    df.dropna(subset=['date_added', 'rating'], inplace=True)

    df['date_added'] = pd.to_datetime(df['date_added'], format='mixed')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month

    return df
