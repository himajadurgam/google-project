from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

with open("documents/institution_info.txt", "r") as file:
    institutional_info = file.read()


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Student Support Assistant is running"}


@app.post("/ask")
def ask_question(data: Question):

    prompt = f"""
You are a Student Support Assistant.

Answer the student's question ONLY using the institutional information below.

Institutional Information:
{institutional_info}

Student Question:
{data.question}

Rules:
1. Use only the institutional information provided.
2. Do not invent information.
3. If the answer is not available, say:
"Sorry, this information is not available in the institutional documents."
4. Give a clear and short answer.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {"answer": response.text}