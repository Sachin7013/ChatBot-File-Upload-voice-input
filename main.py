import os
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from chatbot import chat_with_bot
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF for reading PDFs
from bs4 import BeautifulSoup

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the uploads folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store uploaded file content globally (not ideal for production)
uploaded_content = ""

class ChatRequest(BaseModel):
    message: str

def fetch_and_summarize_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = ' '.join([p.get_text() for p in soup.find_all("p")])
        if not text.strip():
            return "The page does not contain readable text to summarize."
        return text[:1000]  # Limit to first 1000 characters for efficiency
    except requests.exceptions.RequestException as e:
        return f"Could not fetch the URL. Error: {str(e)}"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global uploaded_content
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Save the file locally
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Read the file content
        if file.filename.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        uploaded_content = text  # Store content globally
        return {"message": "File uploaded successfully", "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        message = request.message
        if message.startswith("http://") or message.startswith("https://"):
            url_content = fetch_and_summarize_url(message)
            full_input = f"User input: {message}\nURL content: {url_content}"
        else:
            full_input = f"User input: {message}\nFile content: {uploaded_content}"
        
        response = chat_with_bot(full_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
