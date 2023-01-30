from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit as st
import validators
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 0)

st.title("Sentiment Analysis on Reviews 🚀")

# loading model locally from local file pytorch_model.bin
PATH = 'bert-base-multilingual-uncased-sentiment'
tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)


# loading model only once
@st.cache
def load_model():
    return AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-uncased-sentiment')

model = load_model()


# example link
yelp_url = 'https://www.yelp.com/biz/nylon-coffee-roasters-singapore?osq=coffee'
foursquare_url = 'https://foursquare.com/v/starbucks/4b5305e3f964a520d98c27e3'

choose = st.selectbox(":green[✳ Paste a link that has review contents from selectbox's webpage]", [ 'Yelp', 'Foursquare'])


# input url
webpage_url = st.text_input(":green[✳ Here input a link]", value="")


# validate url format
def Validate_parsedURL():
    if choose == ('Yelp'):
        match = re.search("www\.yelp\.com\/biz", webpage_url)
        if match:
            return True
        else:
            return False
    elif choose == ('Foursquare'):
        match = re.search("foursquare\.com", webpage_url)
        if match:
            return True
        else:
            return False


# show eg link
def showExampleLink():
    if choose == ('Yelp'):
        return yelp_url
    elif choose == ('Foursquare'):
        return foursquare_url


st.write(f' :green[✳ Example link:] {showExampleLink()}')


# get reviews
def get_reviews():
    r = requests.get(webpage_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    if choose == ('Yelp'):
        # from yelp
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class':regex})
    elif choose == ('Foursquare'):
        # from foursquare
        regex = re.compile('.*tipText.*')
        results = soup.find_all('div', {'class':regex})
 
    return [result.text for result in results]


def has_reviews():
    if len(get_reviews()) == 0:
        return False
    else:
        return True


# get torch result
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# get max star reviews
def get_maxStars(df):
    max_val = df['Sentiments'].max()
    df = df[df['Sentiments'] == max_val]
    max_reviews = df['Reviews']
    max_reviews.reset_index()
    return str([r for r in max_reviews])


# main
if Validate_parsedURL():
    if validators.url(webpage_url):
        if st.button('⚙️ Start Analyzing'):
            with st.spinner(text="Analyzing reviews, this could take a moment...🔎"):
                if has_reviews():
                    df = pd.DataFrame(np.array(get_reviews()), columns=['Reviews'])
                    df['Sentiments'] = df['Reviews'].apply(lambda x: sentiment_score(x[:512]))
                    # drop same reviews
                    df = df.drop_duplicates()

                    # histogram setup
                    st.write("")
                    st.subheader("📊 Histogram of Reviews and its Sentiments")
                    st.write("▶ :green[Bars show how much sentiments have on reviews]")
                    fig = px.histogram(df, x = "Reviews", y = "Sentiments", color="Sentiments", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                    # wordcloud setup
                    st.write("")
                    st.write("")
                    st.subheader("🦉Informative Words in 5 Stars⭐ Reviews")
                    maxStarReviews = get_maxStars(df)
                    stopwords = set(STOPWORDS)
                    reviews_wc = WordCloud(width=1200, height=700, background_color="black", max_words=1000, stopwords=stopwords)
                    fig = plt.figure(figsize=(14,14), facecolor='k')
                    reviews_wc.generate(maxStarReviews)
                    plt.imshow(reviews_wc, interpolation='bilinear')
                    plt.tight_layout(pad=0)
                    plt.axis("off")
                    st.write("")
                    st.write(fig)

                    # dataframe setup
                    df['Sentiments'] = df['Sentiments'].apply(lambda x: f"{x} stars" if x > 1 else f"{x} star")
                    st.write("")
                    st.subheader("📝 Table of Reviews and its Sentiments")
                    st.write("🎉 :green[Best sentiment is 5 stars]")
                    st.write(df)

                else:
                    st.warning("⚠ No Review found! Please follow above example link format")
    else:
        st.warning("⚠ Please Enter a valid URL. Don't include any space in the link")
else:
    st.warning("⚠ Please Enter a valid URL.")
