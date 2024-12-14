import streamlit as st
import requests

def is_palindrome(string):
    # URL of the FastAPI endpoint
    url = "http://127.0.0.1:8000/is-palindrome"

    # Send GET request with query parameters
    params = {"input_string": string}
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()  # Return the parsed JSON response
    else:
        return {"error": f"Error: {response.status_code}"}

# Streamlit app
st.title("Palindrome Checker")

st.write("Enter a word, phrase, or sentence to check if it's a palindrome.")

# Input field for the user
user_input = st.text_input("Input your text here:")

# Check if the input is a palindrome
if user_input:
    result = is_palindrome(user_input)
    if "error" in result:
        st.error(result["error"])
    elif result["is_palindrome"]:
        st.success(f"\"{result['original_string']}\" is a palindrome!")
    else:
        st.error(f"\"{result['original_string']}\" is not a palindrome.")
