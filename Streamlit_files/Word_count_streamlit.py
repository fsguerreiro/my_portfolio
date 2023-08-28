import streamlit as st
import wordcloud as wc
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title='Word Count', page_icon=':memo:', layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("""<style>.big-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)

st.title(':memo: Word count tool')

txt = st.text_area('Insert text here:', height=150, placeholder='Type or copy-paste here...')
st.write(f':heavy_check_mark: **This text has {len(txt.split())} words, {len(set(txt.split()))} distinct words, '
         f'{len(txt)} characters and {len("".join(txt.split()))} non-blank characters.**')

# list_exc = {'a', 'an', 'the', 'of', 'with', 'and', 'in', 'on', 'at', 'as'}
ex_box = st.checkbox('Exclude articles, prepositions, pronouns, auxiliary verbs for frequency count')

if st.button('Count words frequency'):
    if txt == '' or len(txt) == 1:
        st.write(':x: Any text bigger than one letter needs to be inserted to make the count.')

    else:
        txt = txt.lower()
        sw = None if ex_box else {'1e6'}
        list_words = wc.WordCloud(stopwords=sw).process_text(txt)

        words = list(list_words.keys())
        freqs = list(list_words.values())

        ddict = {'Words': words, 'Count': freqs}

        df = pd.DataFrame(ddict)

        col1, col2 = st.columns([1, 1], gap='large')
        with col1:
            st.write(':arrow_forward: Click on one of the headers to alter the order.')
            st.dataframe(df, hide_index=True, use_container_width=True)

        with col2:
            st.write(':arrow_forward: The word cloud already excludes articles,prepositions, pronouns, auxiliary verbs')
            wcd = wc.WordCloud(background_color="white").generate(txt)
            plt.imshow(wcd, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            st.pyplot()
