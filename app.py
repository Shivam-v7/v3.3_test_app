import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from fuzzywuzzy import fuzz
import re
from sklearn import tree
import os
from src.account_management import validate
from src.SessionState import *
from src.helpers import *
from mallet import *
import gensim
LOGO_IMAGE = 'images/amrutalogo2.png'
state = get_state()
DEMO_TOPIC_DATA = 'app/news_classification.csv'
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
import pyLDAvis
import pyLDAvis.gensim
import streamlit.components.v1 as components
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# this is the main function in which we define our app
def render_app():
    st.sidebar.title('Amruta OXAI v3.3')
    sample_data_select = st.sidebar.selectbox('Select Sample data:',
                                          ['Topic Modeling','None'])
    pages = {'Topic Modeling': topic_modeling}
    tab = st.sidebar.radio('Select Tab', list(pages.keys()))
    separator = st.sidebar.selectbox('Select separator used in your dataset', ['\t', ',', '|', '$', ' '], 1)
    UPLOADED_DATASET = st.sidebar.file_uploader('Upload your preprocessed dataset in CSV format (Size Limit: 1028mb)', type='csv')
    st.sidebar.image(Image.open(LOGO_IMAGE), width = 300)
    st.sidebar.text('Copyright Amruta Inc. 2021')
    
    df = None
    if UPLOADED_DATASET is not None:
        UPLOADED_DATASET.seek(0)
        sample_data_select = 'None'
        data_load_state = st.text('Loading data...')
        data_to_load = UPLOADED_DATASET
        data_load_state.text('Loading data... done!')
    else:

        if sample_data_select == 'Topic Modeling':
            data_to_load = DEMO_TOPIC_DATA
        else:
            st.info('Please select a sample dataset or upload a dataset.')
            st.stop()
    
        ### LOAD DATA
    if data_to_load:
        try:
            df = load_data(data_to_load, separator)
        except FileNotFoundError:
                st.error('File not found.')
        except:
                st.error('Make sure you uploaded the correct file format.')
    else:
        st.info('Please upload some data or choose sample data.')

    if tab == 'Topic Modeling':
        topic_modeling(df)
    else:
        pass


def topic_modeling(df):

    state.corpus, state.id2word,state.data_lemmatized,state.data_words = data_prep_function(df)

    #lda mallet
    os.environ.update({'MALLET_HOME':r'mallet'})
    mallet_path = 'mallet/bin/mallet'
    st.spinner("Running")
    ldamallet = LdaMallet(mallet_path, corpus=state.corpus, num_topics=20, id2word=state.id2word)

    # Can take a long time to run.
    state.model_list, state.coherence_values = compute_coherence_values(dictionary=state.id2word, corpus=state.corpus, texts=state.data_lemmatized, id2word = state.id2word,num_topics=20, start=2, limit=40, step=6)

    #topic_coherence_df = topic_coherence_check(state.coherence_values)

    #topic_list = list(topic_coherence_df.Num_topics.unique())

    #state.topic_num = st.slider('Select Record index', min_value=0, max_value=-1, value = topic_list,state.topic_num)

    state.optimal_model = state.model_list[3]
    model_topics = state.optimal_model.show_topics(formatted=False)
    st.write("Topics")
    st.write(state.optimal_model.print_topics(num_words=10))    
    
    df_topic_sents_keywords = format_topics_sentences(ldamodel=state.optimal_model, corpus=state.corpus, data=state.data_words)

    st.write("Output")

    df_topic_sents_keywords = dominant_topic_func(state.optimal_model,state.corpus, state.data_lemmatized)
    
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(20)
    
    st.write(df_dominant_topic.head(20))

    st.write("Plot")

    mallet_lda_model= gensim.models.wrappers.ldamallet.malletmodel2ldamodel(state.optimal_model)
    vis = pyLDAvis.gensim.prepare(mallet_lda_model, state.corpus, state.id2word)
    vis = pyLDAvis.gensim.prepare(mallet_lda_model, state.corpus, state.id2word, sort_topics=False)
    st.write('t1')
    try:
        components.html(vis)
    except:
        st.write("part 1 failed")
    st.write('t2')
    try:
        st.markdown(vis, unsafe_allow_html=True)
    except:
        st.write("part 2 failed")    





# def main():
#     if state.user_name == None:
#             title = st.empty()
#             logo = st.empty()
#             text1 = st.empty()
#             text2 = st.empty()
#             usrname_placeholder = st.empty()
#             pwd_placeholder = st.empty()
#             submit_placeholder = st.empty()
#             print('log in page initiated')

#             title.title('Amruta XAI')
#             logo.image(Image.open(LOGO_IMAGE), width=300)
#             text1.text('Copyright Amruta Inc. 2021')
#             text2.text('Beta/Test Version')
#             state.usrname = usrname_placeholder.text_input("User Name", state.usrname if state.usrname else '')
#             state.pwd = pwd_placeholder.text_input("Password", type="password", value=state.pwd if state.pwd else '')
#             state.submit = submit_placeholder.button("Log In")
#             print('log in elements generated')

#             if state.submit:
#                 print(state.submit)
#                 state.validation_status = validate(state.usrname, state.pwd)
#                 if state.validation_status == 'Access Granted':
#                     # store input username to session state
#                     state.user_name = state.usrname
#                     print(state.user_name)

#                     # empty login page elements
#                     title.empty()
#                     logo.empty()
#                     text1.empty()
#                     text2.empty()
#                     usrname_placeholder.empty()
#                     pwd_placeholder.empty()
#                     submit_placeholder.empty()

#                     # start main app
#                     print('main app entered')
#                     render_app()
#                 elif state.validation_status == 'Invalid username/password':
#                     print('Invalid username/password')
#                     st.error("Invalid username/password")
#                 elif state.validation_status == 'Subscription Ended':
#                     print('Your subscription has ended. Please contact us to extend it.')
#                     st.info("Your subscription has ended. Please contact us to extend it.")
#                 # elif:
#                 #     st.error("invalid credentials")
#     else:
#         render_app()

def main():
    render_app()


main()

