import streamlit as st
from rag_engine import (
    load_embed_model, build_chunks, build_index,
    folder_sources, ask_rag,
)

# ---------------- Page config ----------------
DEPT_NAME = "Department of Computer Science and Engineering"  # change to your department
COLLEGE_NAME = "SHRIDEVI INSTITUTE OF ENGINEERING AND TECHNOLOGY"

st.set_page_config(page_title="SIET Department Chatbot", page_icon="🎓", layout="centered")

mykey = st.secrets["GROQ_API_KEYS"]


# ---------------- Cached resources ----------------
@st.cache_resource
def get_embed_model():
    return load_embed_model()


embed_model = get_embed_model()

# ---------------- Session state ----------------
defaults = {
    "chat_hist": [],
    "chunks": [],
    "index": None,
    "doc_names": [],
    "auto_loaded": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def rebuild_knowledge_base(uploaded_files):
    """Combine PDFs from 'college_docs' folder + uploaded PDFs and build FAISS index."""
    sources = folder_sources("college_docs")
    for f in uploaded_files or []:
        sources.append((f.name, f))

    if not sources:
        st.sidebar.warning("No PDFs found. Upload at least one PDF.")
        return

    with st.spinner("Reading PDFs, chunking and embedding..."):
        chunks, errors = build_chunks(sources)
        for e in errors:
            st.sidebar.error(e)
        if not chunks:
            st.sidebar.error("No readable text found in the PDFs.")
            return
        st.session_state.index = build_index(chunks, embed_model)
        st.session_state.chunks = chunks
        st.session_state.doc_names = sorted({c["source"] for c in chunks})


# Auto-load default PDFs from folder once
if not st.session_state.auto_loaded:
    st.session_state.auto_loaded = True
    if folder_sources("college_docs"):
        rebuild_knowledge_base([])

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ Info")

    st.subheader("About the App")
    st.write(
        f"A question-answering chatbot for the **{DEPT_NAME}**. "
        "Ask about courses, syllabus, faculty, FAQs, rules & regulations, "
        "placements and more. Answers come only from the uploaded documents."
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
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "Upload department PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Process documents", use_container_width=True):
        rebuild_knowledge_base(uploaded)

    if st.session_state.doc_names:
        st.success(
            f"{len(st.session_state.doc_names)} file(s), "
            f"{len(st.session_state.chunks)} chunks indexed"
        )
        with st.expander("Indexed files"):
            for n in st.session_state.doc_names:
                st.write("•", n)
    else:
        st.info("No documents indexed yet.")

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

# Show history
for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])
        if show_sources and qa.get("sources"):
            with st.expander("Sources"):
                for s in qa["sources"]:
                    st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
                    st.caption(s["text"][:300] + "...")

question = st.chat_input("Ask anything about the department...")

if question:
    st.session_state.chat_hist.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if st.session_state.index is None:
            answer = "Please upload PDFs in the sidebar and click **Process documents** first."
            hits = []
            st.write(answer)
        else:
            with st.spinner("Generating answer..."):
                try:
                    answer, hits = ask_rag(
                        question,
                        embed_model,
                        st.session_state.chunks,
                        st.session_state.index,
                        api_key=mykey,
                        num_chunks=num_chunks,
                    )
                except Exception as e:
                    answer, hits = f"Error: {e}", []
            st.write(answer)
            if show_sources and hits:
                with st.expander("Sources"):
                    for s in hits:
                        st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
                        st.caption(s["text"][:300] + "...")

    st.session_state.chat_hist.append(
        {"role": "assistant", "content": answer, "sources": hits}
    )
