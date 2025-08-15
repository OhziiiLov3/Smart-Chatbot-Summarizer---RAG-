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



# TEST-FUNCTIONS
if __name__ == "__main__":
    url =  "https://en.wikipedia.org/wiki/OpenAI"  # example URL
    text = fetch_article_text(url)
    print("---- Start of fetched text ----")
    print(text[:1000])  # print first 1000 characters for preview
    print("---- End of fetched text ----")