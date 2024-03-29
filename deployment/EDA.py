import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as ps
from PIL import Image

def run():
# Membuat Title
    st.title('Traffic Net Detection')

    # Membuat Subheader
    st.subheader('Exploratory Data Analysis')

    # Menambahkan deskripsi
    st.write("This model created using Computer Vision Sequential API")
    st.write('Created by Sani - Data Scientist')

    # Membuat garis lurus
    st.markdown('---')

    # Tampilkan accident
    st.image('accident.png', caption='The accident class category is characterized by: An overturned vehicle, Crashed vehicle, Vehicles that collided')
    st.markdown('---')
    # Tampilkan dense
    st.image('dense.png', caption='The Dense Traffic class category is characterized by: Congestion on the road, Congestion is detected by the number of vehicles captured in the image')
    st.markdown('---')
    # Tampilkan sparse
    st.image('sparse.png', caption='The Sparse Traffic class category is characterized by: Slow streets, Not many vehicles are captured in the image, No congestion')
    st.markdown('---')
    # Tampilkan fire
    st.image('fire.png', caption='The Fire class category is characterized by: the presence of fire in the image, there is black smoke caused by the fire, there is a striking red color depicted by the fire')
    st.markdown('---')
    st.write('## Bussiness Insight')
    st.write('### This Traffic Detection model using computer vision can be utilized for:')
    st.write('Accident Detection System')
    st.write('Traffic Prediction System')
    st.write('Transportation Planning System')
    

if __name__ == '__main__':
  run()