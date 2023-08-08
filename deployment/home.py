import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title='Sentiment Analysis', layout='wide', initial_sidebar_state='expanded')

def run():


    # Membuat Sub Header
    st.subheader('Hacktiv8 Phase 2: Milestone 2')

    # Menambahkan Deskripsi
    st.write('App ini dibuat untuk memprediksi sentimen seseorang terhadap tweet chatgpt')

    # Membuat pembatas
    st.markdown('---')

    st.write('Dataset yang digunakan adalah ChatGPT sentiment analysis')

    st.write('Created by: [Dwi Putra Satria Utama](https)')

if __name__ == '__main__':
    run()