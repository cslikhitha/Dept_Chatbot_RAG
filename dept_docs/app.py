import streamlit as st
from rag_engine import (
    load_embed_model, build_chunks, build_index,
    folder_sources, ask_rag,
)

# ---------------- Page config ----------------
DEPT_NAME = "Department of Computer Science and Engineering"  # change to your department
COLLEGE_NAME = "SHRIDEVI INSTITUTE OF ENGINEERING AND TECHNOLOGY"
DOCS_FOLDER = "dept_docs"

st.set_page_config(page_title="SIET Department Chatbot", page_icon="🎓", layout="centered")

mykey = st.secrets["GROQ_API_KEYS"]


# ---------------- Cached resources ----------------
@st.cache_resource
def get_embed_model():
    return load_embed_model()


@st.cache_resource
def get_knowledge_base():
    """Read all PDFs from the college_docs folder, chunk, embed and index them once."""
    sources = folder_sources(DOCS_FOLDER)
    if not sources:
        return None, [], [], []
    chunks, errors = build_chunks(sources)
    if not chunks:
        return None, [], [], errors
    index = build_index(chunks, get_embed_model())
    return index, chunks, sorted({c["source"] for c in chunks}), errors


embed_model = get_embed_model()
with st.spinner("Loading department documents..."):
    index, chunks, doc_names, load_errors = get_knowledge_base()

# ---------------- Session state ----------------
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ Info")

    st.subheader("About the App")
    st.write(
        f"A question-answering chatbot for the **{DEPT_NAME}**. "
        "Ask about courses, syllabus, faculty, FAQs, rules & regulations, "
        "placements and more. Answers come only from the department documents."
    )

    st.subheader("RAG Pipeline")
    st.markdown(
        """
1. **Load** – PDFs are read with `pypdf`
2. **Chunk** – text split into 500-character chunks (100 overlap)
3. **Embed** – `all-MiniLM-L6-v2` creates vectors
4. **Index** – vectors stored in a `FAISS` index
5. **Retrieve** – top-k chunks matched to your question
6. **Generate** – `Groq` LLM answers using only those chunks
        """
    )

    st.divider()
    st.subheader("📄 Knowledge Base")
    if doc_names:
        st.success(f"{len(doc_names)} file(s), {len(chunks)} chunks indexed")
        with st.expander("Indexed files"):
            for n in doc_names:
                st.write("•", n)
    else:
        st.warning(f"No PDFs found in the '{DOCS_FOLDER}' folder.")
    for e in load_errors:
        st.error(e)

    st.divider()
    st.subheader("⚙️ Settings")
    num_chunks = st.slider("Chunks to retrieve (top-k)", 1, 8, 3)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.rerun()

    st.caption("Created by LIKHITHA SK")

# ---------------- Main window ----------------
st.title(COLLEGE_NAME)
st.caption(f"{DEPT_NAME} Chatbot")
st.caption("Ask anything about Courses, FAQs, Rules and Regulations, Placements...")
st.caption("RAG Pipeline with multiple files")


def show_sources_box(hits):
    with st.expander("Sources"):
        for s in hits:
            st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
            st.caption(s["text"][:300] + "...")


# Show history
for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])
        if show_sources and qa.get("sources"):
            show_sources_box(qa["sources"])

question = st.chat_input("Ask anything about the department...")

if question:
    st.session_state.chat_hist.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if index is None:
            answer, hits = f"No documents are loaded. Add PDFs to the '{DOCS_FOLDER}' folder and restart the app.", []
            st.write(answer)
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer, hits = ask_rag(
                        question, embed_model, chunks, index,
                        api_key=mykey, num_chunks=num_chunks,
                    )
                except Exception as e:
                    answer, hits = f"Error: {e}", []
            st.write(answer)
            if show_sources and hits:
                show_sources_box(hits)

    st.session_state.chat_hist.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
