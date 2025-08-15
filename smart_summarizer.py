import os 
from dotenv import load_dotenv
from openai import OpenAI
import requests 
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate



# Step 1: Load api key from .env and initailize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 2: Fetch Article Text
"""
    Fetch and clean the main text of an article from a URL.
    Steps:
    1. Download HTML.
    2. Parse with BeautifulSoup.
    3. Remove scripts, styles, nav, footer, aside, header.
    4. Extract text from main sections.
    5. Clean whitespace.
 """

def fetch_article_text(url):
    #  1. Download HTML.
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        #  2. Parse with BeautifulSoup. 
        soup = BeautifulSoup(response.text, "html.parser")
        # 3. Remove scripts, styles, nav, footer, aside, header.
        for tag in soup(["script","style","noscript", "header", "footer","nav","aside"]):
            tag.decompose()
        # 4.  Extract text from main sections.
        main_content = soup.find("article")
        if main_content:
            text = main_content.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
        # 5. Clean whitespace.
        clean_text = " ".join(text.split())
        return clean_text
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return ""
# Step 3: Use prompt template and OpenAI to summarize piece of text 
"""
    Summarize a piece of text using a prompt template and OpenAI API.
    Steps:
    1. Define the prompt template.
    2. Format the prompt with the input text and chosen style.
    3. Call the OpenAI API to generate a summary.
    4. Extract and return the summary text.
"""
def summarize_text(text, style="concise"):
#   1. Define the prompt template.
#       - 'style' allows dynamic summary types: concise, detailed, bullet points, etc.
#       - 'content' is the actual text we want to summarize
    prompt_template = PromptTemplate(
        input_variables = ["style", "content"],
        template="Summarize the following text in a {style} way: \n\n{content}"
    )
# Step 2:  Format the prompt with the input text and chosen style.
# - Use the first 12,000 characters as a simple token guard
    prompt = prompt_template.format(style=style, content=text[:12000])

# Step 3: Call the OpenAI API to generate a summary.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user", "content":prompt}],
        temperature=0.5,
        max_tokens=250
    )
    return response.choices[0].message.content.strip()


# ------- CLI -------
if __name__ == "__main__":
    choice = input("Enter 'u' for Url or 't' for text: ").strip().lower()
    style = input("Summary style (concise/detailed/bullets): ").strip().lower() or "concise"

    if choice == "u":
        url = input("Enter article URL: ").strip()
        text = fetch_article_text(url)
    else:
        print("Paste your text (end input with empty line):")
        lines = []
        while True:
            line = input()
            if line.strip() == "":
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("Error: No text provided. Exiting.")
    else:
        summary = summarize_text(text, style)
        print("\n Summary:\n")
        print(summary)

# # TEST-FUNCTIONS
# if __name__ == "__main__":
#     url =  "https://en.wikipedia.org/wiki/OpenAI"  # example URL
#     text = fetch_article_text(url)
#     print("---- Start of fetched text ----")
#     print(text[:1000])  # print first 1000 characters for preview
#     print("---- End of fetched text ----")


#     # Generate summary
#     summary = summarize_text(text, style="concise")
#     print("---- Start of summary ----")
#     print(summary)
#     print("---- End of summary ----")