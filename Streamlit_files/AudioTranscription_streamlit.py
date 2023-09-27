import streamlit as st
import whisper
import os
import time

st.set_page_config(page_title='Audio Transcription', page_icon=':loud_sound:', layout='wide')
st.title(':loud_sound: Audio Transcription using Whisper :writing_hand:')
st.divider()

st.write('''Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio 
and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language 
identification. A Transformer sequence-to-sequence model is trained on various speech processing tasks, 
including multilingual speech recognition, speech translation, spoken language identification, and voice activity 
detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a 
single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a 
set of special tokens that serve as task specifiers or classification targets. 
''')

model_dict = {'Size': ['tiny', 'base', 'small', 'medium', 'large'],
              'Required V-RAM': ['~1 GB', '~1 GB', '~2 GB', '~5 GB', '~10 GB'],
              'Relative speed': ['~32x', '~16x', '~6x', '~2x', '1x']}

with st.sidebar:
    st.subheader('Model selection')
    st.write('''There are five model sizes, four with English-only versions, offering speed and accuracy trade-offs.
    Below are the names of the available models and their approximate memory requirements and relative speed. ''')
    st.dataframe(model_dict, hide_index=True, use_container_width=True)
    selected_model_size = st.radio('Select model size: ', model_dict['Size'], index=0)


@st.cache_data
def load_transcribe_audio(model_size, up_audio_file):
    model = whisper.load_model(model_size)
    audio_path = os.path.join(os.path.dirname(__file__), up_audio_file.name)
    result = model.transcribe(audio_path, verbose=False)
    return result['text']


col1, col2 = st.columns([1.25, 1], gap='small')
with col1:
    uploaded_file = st.file_uploader("Choose the audio file:", type=['mp3', 'wav', 'm4a', 'ogg'])

with col2:
    if uploaded_file:
        st.caption('Click the play button to listen.')
        st.audio(uploaded_file, format='audio/wav')

if uploaded_file:
    try:
        start_time = time.time()
        result_text = load_transcribe_audio(selected_model_size, uploaded_file)
        end_time = time.time()
        st.write(f'Time elapsed: {end_time - start_time:.2f} seconds. Here is the transcribed audio:')
        st.markdown(f"> {result_text}")

    except RuntimeError:
        st.error(':x: File not found in the directory. The audio file must be in the same folder as the python code.')
