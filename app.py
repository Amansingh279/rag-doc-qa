import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Local RAG Doc Q&A", page_icon="📄", layout="wide")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

# ---------- Prompts ----------
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question ONLY using the context below.
Use the chat history only to understand what the question refers to.
If the answer is not in the context, reply exactly:
"I don't know based on the provided document."
Always answer in English.

Context:
{context}

Chat history:
{history}

Question: {question}

Answer:"""
)

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the chat history and a follow-up question, rewrite the follow-up
as a standalone question that can be understood without the history.
If it is already standalone, return it unchanged.
Return ONLY the question, nothing else.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""
)

MAP_PROMPT = ChatPromptTemplate.from_template(
    """Summarize the following part of a document in 4-6 concise bullet points.
Keep important names, numbers, skills, dates and projects.

Text:
{text}

Bullet points:"""
)

REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """Below are notes taken from different parts of one document.
Write a clear, well-organized summary of the whole document in English.
Start with one line saying what the document is, then list the key sections
and important details using short bullet points.

Notes:
{text}

Summary:"""
)


# ---------- Helpers ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    # Runs locally on your machine, no API call
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm():
    return ChatGroq(model=LLM_MODEL, temperature=0)


def stream_text(chain_stream):
    """Turn a LangChain stream into plain text pieces for st.write_stream."""
    for chunk in chain_stream:
        if chunk.content:
            yield chunk.content


def format_history(messages, limit=6, max_chars=500):
    recent = messages[-limit:]
    return "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:max_chars]}" for m in recent
    )


# ---------- Core: ingestion ----------
def build_vectorstore(uploaded_file):
    """PDF -> pages -> chunks -> in-memory Chroma. Returns (vectorstore, chunks)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return None, []  # scanned / empty PDF

    vs = Chroma.from_documents(
        chunks,
        embedding=get_embeddings(),
        collection_name=f"doc_{uuid.uuid4().hex[:8]}",  # no persist_directory = in-memory
    )
    return vs, chunks


# ---------- Feature 3: conversation memory ----------
def condense_question(messages, question):
    """Rewrite a follow-up like 'Explain the second one' into a standalone question."""
    if not messages:
        return question
    result = (CONDENSE_PROMPT | get_llm()).invoke(
        {"history": format_history(messages), "question": question}
    )
    return result.content.strip() or question


# ---------- Feature 2: streaming answers ----------
def prepare_answer(vs, question, messages):
    """Returns (text_generator, source_docs). `messages` = history BEFORE this question."""
    standalone = condense_question(messages, question)
    docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(standalone)
    context = "\n\n".join(d.page_content for d in docs)
    chain = ANSWER_PROMPT | get_llm()
    stream = chain.stream(
        {
            "context": context,
            "history": format_history(messages) or "(none)",
            "question": question,
        }
    )
    return stream_text(stream), docs


# ---------- Feature 1: summarize whole document ----------
def prepare_summary(chunks, batch_chars=6000, max_batches=10):
    """Map step: summarize parts of the document. Returns combined notes."""
    batches, current, size = [], [], 0
    for c in chunks:
        text = c.page_content
        if size + len(text) > batch_chars and current:
            batches.append("\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append("\n".join(current))

    if len(batches) > max_batches:  # very long PDF: sample evenly to stay within free limits
        step = len(batches) / max_batches
        batches = [batches[int(i * step)] for i in range(max_batches)]

    if len(batches) == 1:
        return batches[0]

    llm = get_llm()
    partials = [(MAP_PROMPT | llm).invoke({"text": b}).content for b in batches]
    return "\n\n".join(partials)


def stream_summary(notes):
    """Reduce step: final summary, streamed."""
    return stream_text((REDUCE_PROMPT | get_llm()).stream({"text": notes}))


# ---------- Chat export ----------
def build_chat_export(messages, file_name, fmt="md"):
    """Turn the conversation into a .md or .txt string."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    if fmt == "md":
        lines += [f"# Chat with {file_name}", f"_Exported on {stamp}_", ""]
        for m in messages:
            who = "You" if m["role"] == "user" else "Assistant"
            lines += [f"### {who}", m["content"], ""]
    else:
        lines += [f"Chat with {file_name}", f"Exported on {stamp}", "=" * 40, ""]
        for m in messages:
            who = "You" if m["role"] == "user" else "Assistant"
            lines += [f"{who}:", m["content"], ""]
    return "\n".join(lines)


# ---------- Reset ----------
def clear_all():
    vs = st.session_state.get("vs")
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
    for key in ["vs", "chunks", "file_name", "messages", "num_chunks"]:
        st.session_state.pop(key, None)
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1


# ---------- UI ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

with st.sidebar:
    st.title("📄 Document Upload")
    uploaded = st.file_uploader(
        "Upload your PDF here",
        type="pdf",
        key=f"uploader_{st.session_state['uploader_key']}",
    )
    summarize_clicked = st.button("📝 Summarize document")
    st.button("🗑️ Clear Data & Start Over", on_click=clear_all)
    download_slot = st.container()
    st.divider()
    st.subheader("How it works:")
    st.markdown(
        "1. Upload a PDF.\n"
        "2. The system reads and chunks the text.\n"
        "3. It builds a temporary Vector Database.\n"
        "4. You can ask questions or get a summary!"
    )

st.title("Chat with your PDF")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your `.env` file.")
    st.stop()

# Build vector store once per uploaded file
if uploaded and st.session_state.get("file_name") != uploaded.name:
    with st.spinner("Reading the PDF and building the index..."):
        vs, chunks = build_vectorstore(uploaded)
    if vs is None:
        st.warning("No text found in this PDF (it may be scanned). Please try another PDF.")
    else:
        st.session_state.update(
            vs=vs,
            chunks=chunks,
            file_name=uploaded.name,
            num_chunks=len(chunks),
            messages=[],
        )
        st.success(f"Ready! Created {len(chunks)} chunks.")

if "vs" not in st.session_state:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Summarize button
if summarize_clicked:
    st.session_state.messages.append(
        {"role": "user", "content": "Summarize this document."}
    )
    with st.chat_message("user"):
        st.markdown("Summarize this document.")
    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the whole document..."):
                notes = prepare_summary(st.session_state.chunks)
            summary = st.write_stream(stream_summary(notes))
        except Exception as e:
            summary = f"Error: {e}"
            st.error(summary)
    st.session_state.messages.append({"role": "assistant", "content": summary})

# Question box
if q := st.chat_input("Ask anything about your PDF..."):
    history = list(st.session_state.messages)  # history BEFORE this question
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        src = []
        try:
            with st.spinner("Thinking..."):
                stream, src = prepare_answer(st.session_state.vs, q, history)
            ans = st.write_stream(stream)
        except Exception as e:
            ans = f"Error: {e}"
            st.error(ans)
        if src:
            with st.expander("Sources"):
                for d in src:
                    page = d.metadata.get("page", 0) + 1
                    st.caption(f"Page {page}")
                    st.write(d.page_content[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": ans})

# Download chat buttons (filled last so they include the newest messages)
with download_slot:
    if st.session_state.messages:
        base = os.path.splitext(st.session_state.get("file_name", "chat"))[0]
        st.caption("Download chat")
        col1, col2 = st.columns(2)
        col1.download_button(
            "⬇️ .md",
            data=build_chat_export(st.session_state.messages, base, "md"),
            file_name=f"{base}_chat.md",
            mime="text/markdown",
            on_click="ignore",
        )
        col2.download_button(
            "⬇️ .txt",
            data=build_chat_export(st.session_state.messages, base, "txt"),
            file_name=f"{base}_chat.txt",
            mime="text/plain",
            on_click="ignore",
        )