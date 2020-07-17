import streamlit as st
import pandas as pd

@st.cache
def load_data(name):
    return pd.read_pickle('./data/' + name)
def add_commas_to_number(num):
# convert to string with commas and remove decimals from rounding
    return f'{num:,}'.split('.')[0]
