from fastapi import FastAPI, File, UploadFile, Form
from docx import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import numpy as np
import os

app = FastAPI()

# Health check endpoint (for Render /docs visibility)
@app.get("/")
def root():
    return {"message": "FastAPI Docx Bot is live!"}

# Load model and OpenAI client
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_docx(file_like):
    document = Document(file_like)
    return "\n".join([para.text for para in document.paragraphs])

def chunk_text(text, chunk_size=1500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def create_faiss_index(chunks):
    embeddings = [model.encode(chunk) for chunk in chunks]
    arr = np.array(embeddings)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index, arr

def get_similar_chunks(index, chunks, query, k=3):
    q_vec = model.encode(query).reshape(1, -1)
    distances, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]

def get_gpt_response(context, query):
    messages = [
        {"role": "system", "content": "Answer the question using only the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=600
    )
    return res.choices[0].message.content.strip()

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), query: str = Form(...)):
    content = await file.read()
    text = extract_text_from_docx(io.BytesIO(content))
    chunks = chunk_text(text)
    index, _ = create_faiss_index(chunks)
    similar_chunks = get_similar_chunks(index, chunks, query)
    context = " ".join(similar_chunks)
    answer = get_gpt_response(context, query)
    return {"answer": answer}
