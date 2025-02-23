import os
import requests
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
from bs4 import BeautifulSoup  # For web scraping general URLs
import PyPDF2  # For PDF processing
from sentence_transformers import SentenceTransformer
import asyncio
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Load API keys from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Error: GROQ_API_KEY not found in .env file!")

if not YOUTUBE_API_KEY:
    raise ValueError("Error: YOUTUBE_API_KEY not found in .env file!")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

video_data_cache = {}  # Caches transcripts & metadata for faster responses

# Load sentence embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS vector store
vector_db = None

def store_text_in_faiss(text):
    """Splits and stores text in FAISS vector storage."""
    global vector_db
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = text_splitter.create_documents([text])
        vector_db = FAISS.from_documents(documents, embedding_model)
        print("Text stored in FAISS successfully.")
    except Exception as e:
        print(f"Error storing text in FAISS: {str(e)}")

def retrieve_relevant_info(query):
    """Retrieves relevant context from stored FAISS vectors."""
    global vector_db
    if vector_db:
        try:
            docs = vector_db.similarity_search(query, k=2)
            context = " ".join([doc.page_content for doc in docs])
            print(f"Retrieved context: {context}")
            return context
        except Exception as e:
            print(f"Error retrieving info from FAISS: {str(e)}")
    return ""

def extract_video_id(video_url):
    """Extracts the video ID from a YouTube URL"""
    if "watch?v=" in video_url:
        return video_url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[-1].split("?")[0]
    return None

def fetch_youtube_transcript(video_id):
    """Fetches transcript of a YouTube video using YouTubeTranscriptApi"""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text[:5000]  # Limit characters for efficiency
    except (TranscriptsDisabled, NoTranscriptFound):
        return None  # No subtitles available
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def fetch_youtube_metadata(video_id):
    """Fetches video title & description using YouTube API"""
    youtube_api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(youtube_api_url)

    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            snippet = data["items"][0]["snippet"]
            title = snippet["title"]
            description = snippet["description"][:1000]  # Limit description size
            return title, description
        else:
            return None, "No metadata found."
    else:
        return None, f"Error fetching metadata: {response.text}"

def fetch_webpage_content(url):
    """Fetches the main content of a webpage and returns clean text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return f"Error fetching webpage: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")  # Extract paragraphs
        text_content = "\n".join([p.get_text() for p in paragraphs])

        return text_content[:5000]  # Limit characters for efficiency
    except Exception as e:
        return f"Error fetching webpage content: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(pdf_file.file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text[:5000]  # Limit characters for efficiency
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

async def summarize_with_groq(context, query):
    """Summarizes text using Groq model."""
    # This function should be implemented to interact with the Groq model
    # For now, it returns a placeholder response
    return f"Summary based on context: {context} and query: {query}"

@app.post("/chat")
async def chat(request: ChatRequest, req: Request):
    try:
        message = request.message.strip().lower()
        if await req.is_disconnected():
            return {"response": "Client disconnected before response."}

        # Check if the message contains a YouTube URL
        if "youtube.com" in message or "youtu.be" in message:
            video_id = extract_video_id(message)
            if video_id:
                transcript = fetch_youtube_transcript(video_id)
                title, description = fetch_youtube_metadata(video_id)
                if transcript:
                    store_text_in_faiss(transcript)
                if title and description:
                    store_text_in_faiss(f"{title}\n{description}")
                context = retrieve_relevant_info(message)
                response = await summarize_with_groq(context, f"Answer this based on context: {message}")
                return {"response": response}

        context = retrieve_relevant_info(message)
        response = await summarize_with_groq(context, f"Answer this based on context: {message}")
        return {"response": response}
    except asyncio.CancelledError:
        print("Request was cancelled (Client disconnected)")
        return {"response": "Request cancelled."}
    except Exception as e:
        return {"response": f"Internal Server Error: {str(e)}"}

     # Initialize Groq model
llm = ChatGroq(model_name="llama3-8b-8192")

async def summarize_with_groq(context, query):
    """Summarizes text using Groq model."""
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Context: {context}\nQuery: {query}")
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error summarizing with Groq: {str(e)}")
        return "Sorry, I encountered an error while processing your request."   

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handles PDF uploads and stores extracted text in FAISS."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    try:
        print(f"Received file: {file.filename}")
        pdf_text = extract_text_from_pdf(file.file)
        print(f"Extracted text: {pdf_text[:100]}...")  # Print first 100 characters for verification
        store_text_in_faiss(pdf_text)
        return {"message": "PDF uploaded and processed successfully!", "pdf_text": pdf_text}
    except Exception as e:
        print(f"Error uploading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")