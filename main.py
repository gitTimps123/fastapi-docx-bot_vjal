from fastapi import FastAPI, File, UploadFile, Form
from docx import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import numpy as np
import io

app = FastAPI()

# Load model and OpenAI client
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_docx(file_like):
    doc = Document(file_like)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, chunk_size=1500):
    paragraphs = text.split("\n")
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_faiss_index(chunks):
    embeddings = [model.encode(chunk) for chunk in chunks]
    embeddings_array = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index, embeddings_array

def get_similar_chunks(index, chunks, query, k=3):
    query_vec = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]], distances[0]

def get_gpt_response(context, query):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional assistant answering user questions based only on context. "
                "Be formal, accurate, and clear. Use bullet points if helpful."
            )
        },
        {
            "role": "user",
            "content": (
                f"Use the following context to answer the question.\n\n"
                f"--- DOCUMENT CONTEXT START ---\n{context}\n--- DOCUMENT CONTEXT END ---\n\n"
                f"QUESTION: {query}\n\n"
                f"INSTRUCTIONS:\n"
                f"- Start with a one-line summary of your answer\n"
                f"- Answer in about 200 words\n"
                f"- Use bullet points or paragraphs\n"
                f"- If the answer is not found in the context, clearly say so"
            )
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=700,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT error: {str(e)}"

@app.post("/ask")
async def ask_question(file: UploadFile = File(...), query: str = Form(...)):
    try:
        file_bytes = await file.read()
        file_stream = io.BytesIO(file_bytes)
        text = extract_text_from_docx(file_stream)
        chunks = chunk_text(text)
        index, _ = create_faiss_index(chunks)
        similar_chunks, _ = get_similar_chunks(index, chunks, query)
        context = " ".join(similar_chunks)
        answer = get_gpt_response(context, query)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"⚠️ Server error: {str(e)}"}
