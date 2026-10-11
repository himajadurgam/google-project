from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google import genai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

import os
import re
import json
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# Keep the Gemini model that already worked for you
GENERATION_MODEL = "gemini-3.5-flash-lite"

DOCUMENTS_FOLDER = Path("documents")

# Local semantic model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Files used to save our local knowledge index
CACHE_FILE = Path("knowledge_cache.json")
EMBEDDING_FILE = Path("knowledge_embeddings.npy")


# ============================================================
# FASTAPI
# ============================================================

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


# ============================================================
# GLOBAL DATA
# ============================================================

knowledge_chunks = []
knowledge_embeddings = None
embedding_model = None


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

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

            if current:
                current += "\n\n" + paragraph
            else:
                current = paragraph

        else:

            if current:
                chunks.append(current)

            # Handle very large paragraphs
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


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    chunks = []

    if not DOCUMENTS_FOLDER.exists():

        raise RuntimeError(
            "The documents folder does not exist."
        )

    for file_path in sorted(DOCUMENTS_FOLDER.iterdir()):

        # -----------------------------
        # PDF
        # -----------------------------

        if file_path.suffix.lower() == ".pdf":

            try:

                reader = PdfReader(str(file_path))

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    text = page.extract_text()

                    if not text:
                        continue

                    text = clean_text(text)

                    page_chunks = split_text(text)

                    for chunk in page_chunks:

                        chunks.append({
                            "source": file_path.name,
                            "page": page_number,
                            "text": chunk
                        })

            except Exception as e:

                print(
                    f"Could not read {file_path.name}: {e}"
                )

        # -----------------------------
        # TEXT FILE
        # -----------------------------

        elif file_path.suffix.lower() == ".txt":

            try:

                text = file_path.read_text(
                    encoding="utf-8"
                )

                text = clean_text(text)

                text_chunks = split_text(text)

                for chunk in text_chunks:

                    chunks.append({
                        "source": file_path.name,
                        "page": None,
                        "text": chunk
                    })

            except Exception as e:

                print(
                    f"Could not read {file_path.name}: {e}"
                )

    return chunks


# ============================================================
# LOAD LOCAL SEMANTIC MODEL
# ============================================================

def load_embedding_model():

    global embedding_model

    print("\nLoading semantic AI model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Semantic model ready.")


# ============================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    print(
        f"\nCreating semantic embeddings for "
        f"{len(chunks)} document sections..."
    )

    texts = []

    for item in chunks:

        # Include source name because it can help
        # distinguish document context.
        text = (
            f"Document: {item['source']}\n"
            f"{item['text']}"
        )

        texts.append(text)

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings.astype(np.float32)


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================

def save_knowledge_base(chunks, embeddings):

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    np.save(
        EMBEDDING_FILE,
        embeddings
    )

    print("\nKnowledge base saved.")


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_cached_knowledge_base():

    global knowledge_chunks
    global knowledge_embeddings

    if not CACHE_FILE.exists():
        return False

    if not EMBEDDING_FILE.exists():
        return False

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        embeddings = np.load(
            EMBEDDING_FILE
        )

        # Make sure cache matches
        if len(chunks) != len(embeddings):

            print(
                "Cached knowledge base is inconsistent."
            )

            return False

        knowledge_chunks = chunks
        knowledge_embeddings = embeddings

        print(
            f"\nLoaded {len(chunks)} cached knowledge sections."
        )

        return True

    except Exception as e:

        print(
            f"Could not load cache: {e}"
        )

        return False


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():

    global knowledge_chunks
    global knowledge_embeddings

    print("\nLoading institutional documents...")

    # Try existing cache first
    if load_cached_knowledge_base():

        print("Using existing semantic knowledge base.")

        return

    # Load documents
    knowledge_chunks = load_documents()

    print(
        f"Loaded {len(knowledge_chunks)} knowledge chunks."
    )

    if not knowledge_chunks:

        raise RuntimeError(
            "No usable content found in documents folder."
        )

    # Create local embeddings
    knowledge_embeddings = create_embeddings(
        knowledge_chunks
    )

    # Save them
    save_knowledge_base(
        knowledge_chunks,
        knowledge_embeddings
    )

    print(
        f"\nCreated {len(knowledge_chunks)} semantic sections."
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def retrieve_relevant_chunks(
    question,
    top_k=6
):

    # Create embedding for student's question
    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Cosine similarity
    scores = knowledge_embeddings @ query_embedding

    # Get highest scoring sections
    best_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in best_indices:

        results.append({
            "source": knowledge_chunks[index]["source"],
            "page": knowledge_chunks[index]["page"],
            "text": knowledge_chunks[index]["text"],
            "score": float(scores[index])
        })

    return results


# ============================================================
# BUILD CONTEXT FOR GEMINI
# ============================================================

def build_context(retrieved_chunks):

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        if item["page"]:

            location = (
                f"{item['source']}, "
                f"page {item['page']}"
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


# ============================================================
# GENERATE FINAL ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = build_context(
        retrieved_chunks
    )

    prompt = f"""
You are a Student Support Assistant for a college.

Your job is to answer student questions using the
institutional documents provided below.

STUDENT QUESTION:
{question}

INSTITUTIONAL DOCUMENT INFORMATION:
{context}


IMPORTANT INSTRUCTIONS:

1. Answer the student's actual question directly.

2. Use the institutional document information as
   the source of truth.

3. You may reason over the information provided.

4. If the documents provide a rule, range, table,
   procedure, or relationship, apply that information
   to the student's question when appropriate.

5. Example:

   If the document says:
   "80-89 marks = A+ and 9 grade points"

   and the student asks:
   "I got 85 marks. What grade will I get?"

   answer naturally:

   "You would receive an A+ grade with 9 grade points,
   because 85 falls within the 80-89 marks range."

6. Do NOT invent information.

7. Do NOT invent fees, dates, rules, policies,
   procedures, contact details, marks, grades,
   eligibility requirements, or other institutional facts.

8. If the provided documents do not contain enough
   information to answer the question, say:

   "Sorry, this information is not available in the institutional documents."

9. Do not mention:
   - embeddings
   - semantic search
   - retrieval
   - AI models
   - prompts
   - internal systems

10. Do not simply copy large portions of the documents.

11. Explain the answer naturally in your own words.

12. Give enough information to be useful, but avoid
    unnecessary long explanations.

13. If the student asks for steps and the documents
    contain a procedure, present the procedure clearly
    as numbered steps.

14. If the question is ambiguous, use the available
    institutional context to interpret it reasonably.
    Do not invent missing details.

15. Accuracy is more important than sounding confident.

Now provide the best possible answer to the student.
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# REQUEST MODEL
# ============================================================

class Question(BaseModel):

    question: str


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    load_embedding_model()

    build_knowledge_base()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Student Support Assistant is running",
        "semantic_search": True,
        "documents_loaded": len(knowledge_chunks)
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    data: Question
):

    question = data.question.strip()

    if not question:

        return {
            "answer": "Please enter a question."
        }

    try:

        # Find relevant information
        retrieved_chunks = retrieve_relevant_chunks(
            question,
            top_k=6
        )

        # Ask Gemini to formulate the answer
        answer = generate_answer(
            question,
            retrieved_chunks
        )

        return {
            "answer": answer
        }

    except Exception as e:

        print(
            "\nERROR:",
            e
        )

        return {
            "answer": (
                "Sorry, I was unable to process "
                "your question right now."
            )
        }