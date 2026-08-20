import os
import json
import urllib.request
import io
import ssl
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

def expand_query(user_query: str) -> dict:
    if not API_KEY:
        return {"semantic_query": user_query, "object_keywords": [], "ocr_keywords": []}
    
    prompt = f"""
    You are an expert at Video Retrieval and Search Query Expansion.
    The user is looking for a specific scene in a video database.
    The database uses a CLIP model for semantic search (which works best with English descriptions of visual scenes) and exact keyword matching for Objects (e.g. car, person, dog) and OCR (text visible in the video).
    
    Given the user's natural language query (which might be in Vietnamese or English), break it down into 3 parts:
    1. 'semantic_query': Translate to English and optimize for CLIP. Focus on the main visual action/scene. Ignore specific text or numbers. Keep it concise.
    2. 'object_keywords': A list of English keywords for main physical objects mentioned. Keep it simple (e.g. ["car", "red shirt", "man"]). Empty list if none.
    3. 'ocr_keywords': A list of exact text, numbers, or letters the user wants to see written on screen (like license plates, signs). Empty list if none.
    
    User Query: \"{user_query}\"
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return {
            "semantic_query": result.get("semantic_query", user_query),
            "object_keywords": result.get("object_keywords", []),
            "ocr_keywords": result.get("ocr_keywords", [])
        }
    except Exception as e:
        print("Gemini API Error:", e)
        return {"semantic_query": user_query, "object_keywords": [], "ocr_keywords": []}

def create_grid_image(file_ids: list, grid_size=(4, 4), thumb_w=320, thumb_h=180):
    grid_img = Image.new('RGB', (grid_size[0] * thumb_w, grid_size[1] * thumb_h), color='black')
    draw = ImageDraw.Draw(grid_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
        
    for i, file_id in enumerate(file_ids):
        if i >= grid_size[0] * grid_size[1]:
            break
            
        url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w{thumb_w}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=5) as resp:
                img_data = resp.read()
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
                img = img.resize((thumb_w, thumb_h))
                
                row = i // grid_size[0]
                col = i % grid_size[0]
                x = col * thumb_w
                y = row * thumb_h
                
                grid_img.paste(img, (x, y))
                
                draw.rectangle([x, y, x+60, y+50], fill='black')
                draw.text((x+10, y+5), str(i+1), fill='white', font=font)
        except Exception as e:
            print(f"Error downloading {file_id}: {e}")
            
    return grid_img

def rerank_images(query: str, candidates: list) -> list:
    if not API_KEY or not candidates:
        return candidates
        
    top_candidates = candidates[:16]
    file_ids = [c.get("image", {}).get("file_id") or c.get("gdrive_file_id") for c in top_candidates]
    
    print(f"Generating grid image for top {len(file_ids)} candidates...")
    grid_img = create_grid_image(file_ids)
    
    prompt = f"""
    The user is looking for a video scene matching this description: "{query}"
    I have provided a 4x4 grid of candidate images. Each image has a number from 1 to {len(top_candidates)} in the top-left corner.
    Identify WHICH images perfectly match the user's description.
    Return a JSON object with a single key "matches" containing a list of integer numbers corresponding to the matching images.
    If none match perfectly, return an empty list.
    """
    
    try:
        print("Sending grid to Gemini Vision...")
        model = genai.GenerativeModel('gemini-flash-latest', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content([prompt, grid_img])
        result = json.loads(response.text)
        matches = result.get("matches", [])
        print(f"Gemini matches: {matches}")
        
        for num in matches:
            idx = int(num) - 1
            if 0 <= idx < len(top_candidates):
                top_candidates[idx]['score'] += 1000 
                top_candidates[idx]['gemini_verified'] = True
                
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates
    except Exception as e:
        print("Gemini Vision Error:", e)
        return candidates