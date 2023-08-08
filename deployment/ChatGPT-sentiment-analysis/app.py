import streamlit as st
import home
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman:', ('Home', 'Predict'))

if navigation == 'Home':
    home.run()
else:
    prediction.run()