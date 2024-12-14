import streamlit as st

import requests

def translate(prompt):

    input = {
        "prompt": prompt
    }

    # URL of the FastAPI endpoint
    url = "http://127.0.0.1:8000/translate"

    # Send POST request with JSON input
    response = requests.get(url, params=input)

    # Check if the request was successful 
    if response.status_code == 200:
        # Print the response JSON
        output = response.json()
    else:
        output = f"Error: {response.status_code}"

    return output

# Page title
st.set_page_config(page_title='Your 🇬🇧-🇫🇷 Text Translation App')
st.title('Your 🇬🇧-🇫🇷 Text Translation App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('translate_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = translate(txt_input)
            result.append(response)

if len(result):
    st.info(response)