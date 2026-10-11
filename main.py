from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import re
from pathlib import Path

load_dotenv()



if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=GEMINI_API_KEY)

GENERATION_MODEL = "gemini-3.5-flash-lite"
DOCUMENTS_FOLDER = Path("documents")

app = FastAPI(
    title="Student Support Assistant",
    description="Gemini Powered Student Support Assistant",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

knowledge_chunks = []
vectorizer = None
knowledge_matrix = None


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(text, max_chars=1200):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) <= max_chars:
            current = (
                current + "\n\n" + paragraph
                if current else paragraph
            )
        else:
            if current:
                chunks.append(current)

            if len(paragraph) > max_chars:
                words = paragraph.split()
                temp = ""

                for word in words:
                    if len(temp) + len(word) + 1 <= max_chars:
                        temp += (" " if temp else "") + word
                    else:
                        if temp:
                            chunks.append(temp)
                        temp = word

                current = temp
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def load_documents():
    chunks = []

    if not DOCUMENTS_FOLDER.exists():
        raise RuntimeError("The documents folder does not exist.")

    for file_path in sorted(DOCUMENTS_FOLDER.iterdir()):

        if file_path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(file_path))

                for page_number, page in enumerate(
                    reader.pages, start=1
                ):
                    text = page.extract_text()

                    if not text:
                        continue

                    text = clean_text(text)

                    for chunk in split_text(text):
                        chunks.append({
                            "source": file_path.name,
                            "page": page_number,
                            "text": chunk
                        })

            except Exception as e:
                print(f"Could not read {file_path.name}: {e}")

        elif file_path.suffix.lower() == ".txt":
            try:
                text = file_path.read_text(encoding="utf-8")
                text = clean_text(text)

                for chunk in split_text(text):
                    chunks.append({
                        "source": file_path.name,
                        "page": None,
                        "text": chunk
                    })

            except Exception as e:
                print(f"Could not read {file_path.name}: {e}")

    return chunks


def build_knowledge_base():
    global knowledge_chunks
    global vectorizer
    global knowledge_matrix

    print("Loading institutional documents...")

    knowledge_chunks = load_documents()

    print(f"Loaded {len(knowledge_chunks)} knowledge chunks.")

    if not knowledge_chunks:
        raise RuntimeError("No usable content found in documents folder.")

    texts = [
        f"Document: {item['source']}\n{item['text']}"
        for item in knowledge_chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    knowledge_matrix = vectorizer.fit_transform(texts)

    print("Lightweight knowledge retrieval ready.")


def retrieve_relevant_chunks(question, top_k=6):
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector,
        knowledge_matrix
    )[0]

    best_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in best_indices:
        results.append({
            "source": knowledge_chunks[index]["source"],
            "page": knowledge_chunks[index]["page"],
            "text": knowledge_chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


def build_context(retrieved_chunks):
    context_parts = []

    for number, item in enumerate(
        retrieved_chunks, start=1
    ):
        if item["page"]:
            location = (
                f"{item['source']}, page {item['page']}"
            )
        else:
            location = item["source"]

        context_parts.append(
            f"""
SOURCE {number}
Document: {location}

{item["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, retrieved_chunks):

    context = build_context(retrieved_chunks)

    prompt = f"""
You are a Student Support Assistant for a college.

Answer student questions using ONLY the institutional
documents provided below.

STUDENT QUESTION:
{question}

INSTITUTIONAL DOCUMENT INFORMATION:
{context}

IMPORTANT INSTRUCTIONS:

1. Answer the student's actual question directly.

2. Use the institutional document information as the
source of truth.

3. You may reason over the information provided.

4. Apply rules, ranges, tables and procedures from
the documents when appropriate.

5. Do NOT invent information.

6. Do NOT invent fees, dates, rules, policies,
procedures, contact details, marks, grades,
eligibility requirements, or other institutional facts.

7. If the documents do not contain enough information,
say exactly:

"Sorry, this information is not available in the institutional documents."

8. Do not mention retrieval, embeddings, AI models,
prompts or internal systems.

9. Explain naturally in your own words.

10. If the student asks for steps and the documents
contain a procedure, present numbered steps.

11. Accuracy is more important than sounding confident.

Now provide the best possible answer.
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text.strip()


class Question(BaseModel):
    question: str



@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/ask")
def ask_question(data: Question):

    question = data.question.strip()

    if vectorizer is None:
        build_knowledge_base()

    if not question:
        return {"answer": "Please enter a question."}

    try:
        retrieved_chunks = retrieve_relevant_chunks(
            question,
            top_k=6
        )

        answer = generate_answer(
            question,
            retrieved_chunks
        )

        return {"answer": answer}

    except Exception as e:
        print("ERROR:", e)

        return {
            "answer":
            "Sorry, I was unable to process your question right now."
        }
