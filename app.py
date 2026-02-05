import streamlit as st
import nltk
import pickle
import re
from nltk.corpus import stopwords

# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl' , 'rb'))

def clean_text(news):
    news = news.lower()
    news = re.sub(r'[^a-z ]','',news)
    words=news.split()
    return " ".join([w for w in words if w not in stop_words])

# UI 
st.title("FAKE NEWS DETECTION SYSTEM")
st.write("Enter the news article to check wheather it's real or fake")

news = st.text_area("News Text")

if st.button("predict"):
    if news.strip() == "":
        st.warning("Please Enter the news!")
    else:
        clean = clean_text(news)
        vector = vectorizer.transform([clean])
        prediction= model.predict(vector)

        if prediction[0] == 1:
             st.error("❌ Fake News")
        else:
             st.success("✅ Real News")