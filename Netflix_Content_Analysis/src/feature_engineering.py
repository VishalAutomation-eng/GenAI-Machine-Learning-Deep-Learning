def add_content_age(df):
    df['age_on_netflix'] = df['year_added'] - df['release_year']
    return df[df['age_on_netflix'] >= 0]
