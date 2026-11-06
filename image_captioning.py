import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO
import os

# 1. Helper function to load images from a live web URL
def load_image_from_url(url):
    try:
        print(f"Fetching image from URL: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def generate_caption_from_url(image_url):
    print("Loading Pre-trained Vision-Language Transformer (BLIP)...")
    
    # Load processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load image from the provided URL
    image = load_image_from_url(image_url)
    if image is None:
        return
    
    # Preprocess image for the transformer
    inputs = processor(image, return_tensors="pt").to(device)
    
    print("Generating caption...")
    # Generate text tokens using the autoregressive decoder
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    print("\n--- Transformer Model Results ---")
    print(f"Image URL: {image_url}")
    print(f"Generated Caption: '{caption}'")
    
    return caption

if __name__ == "__main__":
    # You can change this URL to test any image on the internet!
    sample_image_url = "https://ethnasia.com/cdn/shop/articles/sean-o-KMn4VEeEPR8-unsplash_edited.jpg?v=1621585619&width=1920"
    
    generate_caption_from_url(sample_image_url)