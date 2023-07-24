import streamlit as st
import numpy as np
import pandas as pd


st.set_page_config(page_title='My first app')
st.write('# My first page')

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

df = pd.DataFrame(
   np.random.randn(10, 5),
   columns=(f'col {i}' for i in range(5)))

st.table(df)
