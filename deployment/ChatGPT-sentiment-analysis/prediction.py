import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Load Model
model_lstm = load_model('best_model')

# Load the dictionary slang/abbreviations menjadi kalimat standar (baku)
slang_to_standard = {}
with open('chatwords.txt', 'r') as file:
    for line in file:
        words = line.strip().split()
        slang = words[0].lower()
        standard = ' '.join(words[1:]).lower()
        slang_to_standard[slang] = standard

stop_words = set(stopwords.words('english'))

lem = WordNetLemmatizer()
stemmer = PorterStemmer()

def is_alpha(word):
    for part in word.split('-'):
        if not part.isalpha():
            return False
    return True

def tweets_proses(tweets):
    # Mengubah Teks ke Lowercase
    tweets = tweets.lower()

    # Menghilangkan Mention
    tweets = re.sub("@[A-Za-z0-9_]+", " ", tweets)

    # Menghilangkan Hashtag
    tweets = re.sub("#[A-Za-z0-9_]+", " ", tweets)

    # Menghilangkan \n
    tweets = re.sub(r"\\n", " ", tweets)

    # Menghilangkan Whitespace
    tweets = tweets.strip()

    # Menghilangkan Link
    tweets = re.sub(r"http\S+", " ", tweets)
    tweets = re.sub(r"www.\S+", " ", tweets)

    # Menghilangkan yang Bukan Huruf seperti Emoji, Simbol Matematika (seperti μ), dst
    tweets = re.sub("[^A-Za-z\s']", " ", tweets)

    # Menghilangkan RT
    tweets = re.sub("rt", " ", tweets)

    # Melakukan Tokenisasi
    tokens = word_tokenize(tweets)

    # Memecah teks menjadi token (kata-kata) menggunakan word_tokenize dan menyimpan kata-kata yang hanya berisi huruf tanda _
    words = [word for word in word_tokenize(tweets) if is_alpha(word)]

    # Melakukan lemmatisasi pada setiap kata untuk mengubahnya ke bentuk kata dasar
    words = [lem.lemmatize(word) for word in words]

    # Membuang kata-kata yang termasuk dalam stopwords (kata-kata umum yang sering tidak memberikan banyak informasi).
    words = [w for w in words if not w in stop_words]

    # Mengganti slang and abbreviations dengan kalimat baku
    words = [slang_to_standard[word.lower()] if word.lower() in slang_to_standard else word for word in words]

    # Stemming menggunakan NLTK Porter Stemmer
    #words = [stemmer.stem(word) for word in words]

    # Joining the words back to form the processed text
    text = " ".join(words)

    return text

# Streamlit app function
def run():
    # Menambahkan Deskripsi Form
    st.write('Page ini berisi model untuk memprediksi jenis sentimen tweet chatgpt')

    with st.form(key='form_tweet'):
        st.markdown('### **Tweet**')
        tweet_text = st.text_input('', value='')
        submitted = st.form_submit_button('Predict')

    # Membuat Dataframe
    data_inf = {
        'tweet_text': tweet_text
    }
    data_inf = pd.DataFrame([data_inf])

    if submitted:
        # Preprocessing Data Inference
        data_inf['tweet_processed'] = data_inf['tweet_text'].apply(lambda x: tweets_proses(x))

        # Prediksi jenis tweet
        y_inf_pred = np.argmax(model_lstm.predict(data_inf['tweet_processed']), axis=-1)

        # Membuat fungsi untuk return result prediksi
        if y_inf_pred[0] == 0:
            result = 'bad'
        elif y_inf_pred[0] == 1:
            result = 'good'
        else:
            result = 'neutral'
        st.write('# Sentiment Prediction : ', result)

if __name__ == '__main__':
    run()
