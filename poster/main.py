from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from diffusers import StableDiffusionPipeline
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_id = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
gen_ai_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4, google_api_key='AIzaSyARn_PcqweM5MXHxYaIWGQcf-BDJMP1bDw')

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)

class ProductData(BaseModel):
    product_description: str
    tagline_description: str

def generate_poster(prompt: str) -> Image.Image:
    image = pipe(prompt).images[0]
    return image

def generate_catchy_text(product_type: str) -> str:
    prompt_template = f"""
    Generate a short 3-4 words catchy text or slogan for the {product_type} which displays in the advertisement poster.
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["product_type"])
    chain = LLMChain(llm=gen_ai_model, prompt=prompt)
    catchy_text = chain.run(product_type=product_type)
    return catchy_text

def add_text_outside_box(image_pil: Image.Image, catchy_text: str) -> Image.Image:
    font_path = "Raleway-Bold.ttf"  
    try:
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"Could not load font: {font_path}")
        return image_pil

    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width = image.shape[:2]
    text_y = image_height // 2

    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    text_x = image_width // 2 - 100
    draw.text((text_x, text_y), catchy_text, font=font, fill=(255, 255, 255))

    return pil_image

async def generate_poster_in_background(data: ProductData, background_tasks: BackgroundTasks):
    poster = generate_poster(prompt=data.product_description)
    catchy_text = generate_catchy_text(product_type=data.tagline_description)
    final_poster = add_text_outside_box(poster, catchy_text)
    
    output_image_path = "generated_poster.png"
    final_poster.save(output_image_path)
    background_tasks.add_task(save_and_return_image, output_image_path)

# Separate task to save and return the image
async def save_and_return_image(output_image_path: str):
    return FileResponse(output_image_path, media_type='image/png', filename='generated_poster.png')

@app.post("/generate_poster/")
async def create_poster(data: ProductData, background_tasks: BackgroundTasks):
    # Perform the poster generation in background
    await generate_poster_in_background(data, background_tasks)
    return {"message": "Poster generation is in progress"}