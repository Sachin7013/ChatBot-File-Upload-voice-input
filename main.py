import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from chatbot import chat_with_bot
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF for reading PDFs

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
        # Include uploaded file content in the chatbot query
        full_input = f"User input: {request.message}\nFile content: {uploaded_content}"
        response = chat_with_bot(full_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
