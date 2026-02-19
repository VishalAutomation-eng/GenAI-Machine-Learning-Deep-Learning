from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(df):
    text = ' '.join(df['description'])
    wc = WordCloud(width=800, height=400, background_color='black').generate(text)

    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
