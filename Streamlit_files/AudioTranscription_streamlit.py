import streamlit as st
import whisper
from tempfile import NamedTemporaryFile

st.set_page_config(page_title='Audio Transcription', layout='wide')
st.title('Audio Transcription')
#
#
# st.write(uploaded_file)
#st.write(uploaded_file.name)
# uploaded_file = r'C:\Users\ferna\Desktop\Python e Ciencia de dados\Arquivos de python\my_portfolio\Streamlit_files\sample1.wav'

#


col1, col2 = st.columns([1, 1], gap='large')
with col1:
    uploaded_file = st.file_uploader("Choose a file", type=['mp3', 'wav', 'm4a', 'ogg'])
    mode = st.radio('Select mode: ', ['tiny', 'base', 'small'], index=1)
    model = whisper.load_model(mode)
    audio_bytes = uploaded_file.read()

with col2:
    if uploaded_file:
        st.audio(audio_bytes, format='audio/wav')
        st.write(uploaded_file.name)
        result = model.transcribe(uploaded_file.name)
        st.text_area('Transcribed audio:', result["text"])

