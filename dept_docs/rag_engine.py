import faiss
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# ---------- Embedding model ----------
def load_embed_model(model_name=EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)


# ---------- Load PDF (path or uploaded file object) ----------
def load_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = [p.extract_text() or "" for p in reader.pages]
    full_text = "\n".join(pages).strip()
    if not full_text:
        raise ValueError("No readable text found")
    return full_text


# ---------- Chunking ----------
def create_chunks(full_text, chunksize=500, overlap=100):
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ---------- Build chunk list from many sources ----------
def build_chunks(sources):
    """
    sources: list of (file_name, pdf_path_or_file_object)
    returns: (chunks, errors)
      chunks = [{"text": ..., "source": file_name}, ...]
    """
    chunks, errors = [], []
    for name, pdf in sources:
        try:
            text = load_pdf(pdf)
            for c in create_chunks(text):
                chunks.append({"text": c, "source": name})
        except Exception as e:
            errors.append(f"{name}: {e}")
    return chunks, errors


def folder_sources(folder="college_docs"):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [(p.name, str(p)) for p in sorted(folder.glob("*.pdf"))]


# ---------- Embedding + FAISS index ----------
def build_index(chunks, embed_model):
    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")
    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ---------- Retrieve ----------
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if num_chunks < 1:
        raise ValueError("Request at least one chunk")
    q = embed_model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, ids = index.search(q, min(num_chunks, len(chunks)))
    hits = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1:
            hits.append({
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "score": float(score),
            })
    return hits


# ---------- Ask ----------
def ask_rag(question, embed_model, chunks, index, api_key,
            num_chunks=3, model_name="openai/gpt-oss-120b", temperature=0.3):
    if not api_key:
        raise ValueError("API key missing")

    hits = retrieve_content(question, embed_model, chunks, index, num_chunks)
    background = "\n\n".join(h["text"] for h in hits)

    prompt = (
        "You are a professional AI assistant for a college department.\n"
        "Answer the user question using ONLY the document background below.\n"
        "Do not invent or assume facts.\n"
        "If the information is not available, simply say: Information Missing.\n\n"
        f"Document Background:\n{background}\n\n"
        f"User Question: {question}\n"
        "Answer:"
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content, hits
