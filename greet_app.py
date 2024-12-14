import streamlit as st
import requests

def greet_user(name):
    """Fetch a greeting message from the API."""
    url = "http://127.0.0.1:8000/greet"
    params = {"name": name}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error: {response.status_code}"}

# Streamlit app
st.title("Greeting App")

# Greeting Section
st.header("Greeting")
st.write("Enter your name to get a personalized greeting.")
name_input = st.text_input("Input your name:")

if name_input:
    greeting_result = greet_user(name_input)

    if "error" in greeting_result:
        st.error(greeting_result["error"])
    else:
        st.success(greeting_result["message"])
