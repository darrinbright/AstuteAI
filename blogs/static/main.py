import os
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re
import nltk
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from googletrans import Translator  

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlogInput(BaseModel):
    blog_input: str
    lang_choice: int  

def language(choice):
    lang_map = {
        1: 'hi',  # Hindi
        2: 'bn',  # Bengali
        3: 'te',  # Telugu
        4: 'mr',  # Marathi
        5: 'ta',  # Tamil
        6: 'ur',  # Urdu
        7: 'gu',  # Gujarati
        8: 'kn',  # Kannada
        9: 'ml',  # Malayalam
        10: 'en'  # English
    }
    return lang_map.get(choice, 'en')

def translateentolang(text, target_lang):
    source_lang = 'en'
    translator = Translator()
    translated_text = translator.translate(text, src=source_lang, dest=target_lang)
    return translated_text.text

def load_llm(max_tokens):
    try:
        logger.info("Initializing ChatGenerativeGemini model with Google API key.")
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
        return llm
    except Exception as e:
        logger.error(f"Error loading ChatGenerativeGemini model: {str(e)}")
        raise

def get_src_original_url(query):
    url = 'https://api.pexels.com/v1/search'
    headers = {
        'Authorization': "Ura2E4Y3bRGkVpRMeN5R53GS62WQ1bhVjcntmYod8Ima1diEcS3NWN8l",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
    }
    params = {'query': query, 'per_page': 1}
    try:
        logger.info(f"Sending request to Pexels API with query: {query}")
        response = requests.get(url, headers=headers, params=params)
        logger.info(f"Received response from Pexels API with status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            photos = data.get('photos', [])
            if photos:
                logger.info("Photo found, returning image URL and description")
                return photos[0]['src']['original'], photos[0]['alt']
            else:
                logger.warning("No photos found for the given query.")
                raise HTTPException(status_code=404, detail="No photos found for the given query.")
        elif response.status_code == 403:
            logger.error("403 Forbidden: Check your API key and permissions.")
            raise HTTPException(status_code=403, detail="Forbidden: Check your API key and permissions.")
        else:
            logger.error(f"Pexels API returned error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        logger.error(f"Error fetching image from Pexels API: {str(e)}")
        raise

def clean_content(content):
    return re.sub(r'\*+', '', content)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(f"{word}" for word in filtered_words)

def extract_keywords(text, n=10):
    vectorizer = TfidfVectorizer(max_features=n)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out().tolist()
    return [f"{keyword}" for keyword in keywords]

@app.post("/generate_blog")
def generate_blog(request_body: BlogInput):
    blog_input = request_body.blog_input
    lang_choice = request_body.lang_choice

    logging.info(f"Route has received the data: {blog_input}")
    image_url, image_description = get_src_original_url(blog_input)
    
    prompt_template_str = (
        "You are a digital marketing and SEO expert. Write a blog post consisting of 3 paragraphs inspired by the following description: "
        "{blog_description}. The blog should be around 1500 words long, and it should be informative and engaging."
    )

    prompt_template = PromptTemplate.from_template(prompt_template_str)
    llm = load_llm(max_tokens=600)
    
    formatted_prompt = f"You are a digital marketing and SEO expert. Write a blog post consisting of 3 paragraphs inspired by the following description: {blog_input}. The blog should be around 1500 words long with 3 paragraphs, and it should be informative and engaging."
    
    try:
        logger.info(f"Sending the formatted prompt to the model: {formatted_prompt}")
        result = llm.invoke(input=formatted_prompt)  
        logger.info(f"Full result from model: {result}")

        blog_content = clean_content(result.content)  

    except Exception as e:
        logger.error(f"Error generating blog content: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate blog content.")

    processed_content = preprocess_text(blog_content)
    keywords = extract_keywords(processed_content)
    logger.info(f"Extracted Keywords: {keywords}")

    target_lang = language(lang_choice)
    if target_lang != 'en':
        blog_content = translateentolang(blog_content, target_lang)

    title = f"{blog_input.title()}"

    if blog_content:
        return JSONResponse(content={
            "title": title,
            "blog_content": blog_content,
            "image": image_url,
            "image_alt": image_description,
            "keywords": keywords  
        })
    else:
        raise HTTPException(status_code=500, detail="Your article couldn't be generated!")
