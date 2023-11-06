import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd 
import altair as alt
# from sklearn.neighbors import NearestNeighbors

import settings

# hide footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# hide footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title('CSV File Explorer')

# Load data
data = pd.read_csv('patients.csv') 

# Display statistics
col1, col2 = st.columns(2)

with col1:
   st.header('Statistics')
   st.write(data.describe())

with col2:
   st.header('Visuals')   
   chart = alt.Chart(data).mark_point().encode(
      x='anchor_age',
      y='count()'
   )
   st.altair_chart(chart)

# # Display statistics
# st.header('Data Statistics')
# st.write(data.describe())

# # Display graph 

# st.header('Data Graph')
# chart = alt.Chart(data).mark_point().encode(
#     x='anchor_age',
#     y='count()'
# )
# st.altair_chart(chart)

# Display similar entries
st.header('Similar Entries')
# selected_entry = st.selectbox('Select an entry', data['id'])

# nbrs = NearestNeighbors(n_neighbors=5).fit(data.drop('id', axis=1)) 
# distances, indices = nbrs.kneighbors(data.loc[data['id'] == selected_entry, :].drop('id', axis=1))

# similar_entries = data.loc[indices[0], :]
# st.write(similar_entries)

col1, col2, col3 = st.columns(3)
if col2.button('Inspect EHR Data with Medical ChatBot'):
    switch_page('chatbot')