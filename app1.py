import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from fuzzywuzzy import fuzz
import re
from sklearn import tree

LOGO_IMAGE = 'images/amrutalogo2.png'
file_path_1 = 'app/consolidated_individuals.csv'
file_path_2 = 'app/consolidated_entities.csv'
file_path_3 = 'app/FC_dataset.csv'
model_path = 'app/rf_model.pkl'
shap_0 = 'app/shap_0.png'
shap_1 = 'app/shap_1.png'
shap_2 = 'app/shap_2.png'

def load_model(model):
    loaded_model = pickle.load(open(model_path, 'rb'))
    return loaded_model

def person_token_sort_match_function(input_string):
    c_df = pd.read_csv(file_path_1)
    c_df['Clean_name'] = c_df['Name'].apply(lambda x: re.sub(r'"','',x))
    best_match_score = 0
    best_match_name = ""
    for i in c_df['Clean_name']:
        Ratio = fuzz.token_sort_ratio(i.lower(),input_string.lower())
        if Ratio>best_match_score:
            best_match_score = Ratio
            best_match_name = i
    return [best_match_score,best_match_name]
def entity_token_sort_match_function(input_string):
    c_df = pd.read_csv(file_path_2)
    c_df['Clean_name'] = c_df['Name'].apply(lambda x: re.sub(r'"','',x))
    best_match_score = 0
    best_match_name = ""
    for i in c_df['Clean_name']:
        Ratio = fuzz.token_sort_ratio(i.lower(),input_string.lower())
        if Ratio>best_match_score:
            best_match_score = Ratio
            best_match_name = i
    return [best_match_score,best_match_name] 

#@st.cache(allow_output_mutation=True)
def entity_check_func():
    st.header("Entity Match & Group assignment")
    entity_type = st.selectbox('Investor Type', ('Individual', 'Legal Entity'))
    user_input = st.text_input("Enter the name")
    
    check_hm = st.checkbox('Perform Entity Check')
    if check_hm:
        st.spinner('Running Matching Engine')
        if entity_type == 'Individual':
            func_result1 = person_token_sort_match_function(user_input)
            #st.write('Restricted List')
            st.write('The input name matched %d percent with Restricted List with the name %s .'%(func_result1[0],func_result1[1]))
        elif entity_type == 'Legal Entity':
            #st.write('R List')
            func_result2 = entity_token_sort_match_function(user_input)
            st.write('The input name matched %d percent with Restricted List with the name %s .'%(func_result2[0],func_result2[1]))
        

def risk_score_func():
    st.header("Risk Score Assessment")
    entity_type = st.selectbox('Investor Type', ('Individual', 'Legal Entity'))
    if entity_type == 'Individual':
        rscore = 0

        individual_type = st.selectbox('Individual Type', ['Restricted Investor', 'Low-Risk Investor'])
        if individual_type == 'Restricted Investor':
            rscore = rscore + 500
        elif individual_type == 'Low-Risk Investor':
            rscore = rscore + 0

        fatf = st.selectbox('Under FATF Jurisdiction?', ['Yes', 'No'])
        if fatf == 'Yes':
            fname = st.radio('Is first name present?', ['Yes', 'No'])
            if fname == 'Yes':
                rscore = rscore + 0
            elif fname == 'No':
                rscore = rscore + 100

            lname = st.radio('Is last name present?', ['Yes', 'No'])
            if lname == 'Yes':
                rscore = rscore + 0
            elif lname == 'No':
                rscore = rscore + 100

            address = st.radio('Is an address present?', ['Yes', 'No'])
            if address == 'Yes':
                rscore = rscore + 0
            elif address == 'No':
                rscore = rscore + 100

            ssn_taxid = st.radio('Is a SSN/Tax Identification number present?', ['Yes', 'No'])
            if ssn_taxid == 'Yes':
                rscore = rscore + 0
            elif ssn_taxid == 'No':
                rscore = rscore + 100

        elif fatf == 'No':
            fname = st.radio('Is first name present?', ['Yes', 'No'])
            if fname == 'Yes':
                rscore = rscore + 0
            elif fname == 'No':
                rscore = rscore + 100

            lname = st.radio('Is last name present?', ['Yes', 'No'])
            if lname == 'Yes':
                rscore = rscore + 0
            elif lname == 'No':
                rscore = rscore + 100

            address = st.radio('Is an address present?', ['Yes', 'No'])
            if address == 'Yes':
                rscore = rscore + 0
            elif address == 'No':
                rscore = rscore + 100

            ssn_taxid = st.radio('Is a SSN/Tax Identification number present?', ['Yes', 'No'])
            if ssn_taxid == 'Yes':
                rscore = rscore + 0
            elif ssn_taxid == 'No':
                rscore = rscore + 100

            pass_util = st.radio('Is a passport or utility bill present?', ['Yes', 'No'])
            if pass_util == 'Yes':
                rscore = rscore + 0
            elif pass_util == 'No':
                rscore = rscore + 100

    elif entity_type == 'Legal Entity':
        rscore = 0

        fatf = st.selectbox('Under FATF Jurisdiction?', ['Yes', 'No'])
        if fatf == 'Yes':
            id_require = st.radio('Does the identification include articles of incorporation, government issued license, or a partnership agreement?', ['Yes', 'No'])
            if id_require == 'Yes':
                rscore = rscore + 0
            elif id_require == 'No':
                rscore = rscore + 250

        elif fatf == 'No':
            dod = st.radio('Does the entity have duly organized officers?', ['Yes', 'No'])
            if dod == 'Yes':
                rscore = rscore + 0
            elif dod == 'No':
                rscore = rscore + 250

            so = st.radio('Does the entity have senior officers?', ['Yes', 'No'])
            if so == 'Yes':
                rscore = rscore + 0
            elif so == 'No':
                rscore = rscore + 250

            peo = st.radio('Does the entity have principal equity officers?', ['Yes', 'No'])
            if peo == 'Yes':
                rscore = rscore + 0
            elif peo == 'No':
                rscore = rscore + 250

            aml = st.radio('Does the entity have AML certification that none of its directors, officers or investors are prohibited?', ['Yes', 'No'])
            if aml == 'Yes':
                rscore = rscore + 0
            elif aml == 'No':
                rscore = rscore + 250

    result = ""
    if st.button("Calculate Score"):
        result = rscore
        st.write('Risk Assessment Score:', result)
        
        
def Crime_type_func():
    df = pd.read_csv(file_path_3)
    st.write("Data Exploration")
    st.write(df.head(5))
    # plot tree
    st.write('Model Performance')
    st.image(Image.open(shap_0))
    
    st.write('Model Explainabilty')
    st.image(Image.open(shap_1))
    st.image(Image.open(shap_2))
    



# this is the main function in which we define our app
def main():
    st.sidebar.title('Amruta OXAI Demo')

    pages = {
    'Entity Check': entity_check_func,
    'Risk Score': risk_score_func,
    'Crime type': Crime_type_func
    }
    tab = st.sidebar.radio('Select Tab', list(pages.keys()))
    st.sidebar.image(Image.open(LOGO_IMAGE), width = 300)
    st.sidebar.text('Copyright Amruta Inc. 2021')



    if tab == 'Entity Check':
        entity_check_func()
    elif tab == 'Risk Score':
        risk_score_func()
    elif tab=='Crime type':
        Crime_type_func()










main()
