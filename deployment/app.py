import streamlit as st
import prediction
import EDA

navigation = st.sidebar.selectbox('Choose Page:',('Traffic Net Detection','EDA'))

if navigation == 'EDA':
    EDA.run()
else:
    prediction.run()