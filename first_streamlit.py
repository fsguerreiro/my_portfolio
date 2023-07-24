import streamlit as st
import numpy


st.set_page_config(page_title='My first app')
st.write('# My first page')

col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")
