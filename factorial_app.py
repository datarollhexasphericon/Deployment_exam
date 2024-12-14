import streamlit as st
import requests

def get_factorial(number):
    """Fetch the factorial of a number from the API."""
    url = "http://127.0.0.1:8000/factorial"
    params = {"number": number}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error: {response.status_code}"}

# Streamlit app
st.title("Factorial Calculator")

st.write("Enter a number to calculate its factorial.")

# Input field for the user
user_input = st.text_input("Input a number:", value="0")

# Check if the input is valid and calculate the factorial
if user_input:
    try:
        number = int(user_input)
        result = get_factorial(number)

        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"The factorial of {result['number']} is {result['factorial']}.")
    except ValueError:
        st.error("Please enter a valid integer.")
