from fastapi import FastAPI
from transformers import pipeline

# Create the FastAPI application
app = FastAPI()

# 1. Basic endpoint that returns a greeting
@app.get("/greet")
def greet(name: str):
    return {"message": f"Hello, {name}! Welcome to my API."}

# 2. Endpoint to calculate the factorial of a number
@app.get("/factorial")
def calculate_factorial(number: int):
    if number < 0:
        return {"error": "Factorial is not defined for negative numbers."}
    factorial = 1
    for i in range(1, number + 1):
        factorial *= i
    return {"number": number, "factorial": factorial}

# 3. Hugging Face endpoint for summary
@app.get("/summarize")
def generate_text(prompt):
    summarizer = pipeline("summarization")
    response = summarizer(prompt, min_length=5, max_length=20)
    return response[0]["summary_text"]

# 4. Hugging Face endpoint for translation
@app.get("/translate")
def translate(prompt):
    en_fr_translator = pipeline("translation_en_to_fr")
    response = en_fr_translator(prompt)
    return response[0]["translation_text"]


# 5. Endpoint to check if a string is a palindrome
@app.get("/is-palindrome")
def is_palindrome(input_string: str):
    sanitized = input_string.replace(" ", "").lower()
    is_palindrome = sanitized == sanitized[::-1]
    return {"original_string": input_string, "is_palindrome": is_palindrome}