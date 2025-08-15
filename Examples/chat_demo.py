import os 
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load environment variables from .env
load_dotenv()

# Step 2: Get API key 
api_key = os.getenv("OPENAI_API_KEY")

# Step 3: Initialize OpenAI client 
client = OpenAI(api_key=api_key)


# example Basic API call - from OpenAI docs using responses endpoint 
# What it does -  It sends a request to the OpenAI model (`gpt-4o-mini`  in this case) and asks it to generate some output.
# response = client.responses.create(
#     model="gpt-4o-mini",  # choose model here 
#     input="Wrtie a one-sentence bedtime story about a English bulldog named Zooty."
# )
# print(response.output_text)

# Chat completion example with `messages` is **how you structure the conversation** between the system, user, and assistant.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system", "content": "You are a helpful Python coding assistant."},
        {"role":"user", "content": "Write a Python function to reverse a string."},
    ]
)

# Print AI's reply
print(response.choices[0].message.content)