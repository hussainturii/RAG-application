import os
import tempfile

import chromadb
import ollama
import pandas as pd
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


# ---------------- SYSTEM PROMPT ----------------
system_prompt = """
You are an AI assistant tasked with providing answers based ONLY on the given context.

CRITICAL RULES:
- If the context does NOT contain information to answer the question, respond EXACTLY with:
  "I cannot find information about this topic in the document."
- Do NOT use external knowledge
- Do NOT make assumptions
- Only answer if information is clearly present in the context

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state: "I cannot find information about this topic in the document."

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


# ---------------- DOCUMENT PROCESSING ----------------
def process_document(uploaded_file: UploadedFile) -> list[Document]:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile("wb", suffix=f".{file_extension}", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        if file_extension == "pdf":
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()

        elif file_extension == "txt":
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"page": 0})]

        elif file_extension == "docx":
            import docx2txt
            text = docx2txt.process(temp_path)
            docs = [Document(page_content=text, metadata={"page": 0})]

        elif file_extension in ["xlsx", "xls"]:
            excel = pd.ExcelFile(temp_path)
            all_text = []
            for sheet in excel.sheet_names:
                df = pd.read_excel(temp_path, sheet_name=sheet)
                all_text.append(f"\n--- Sheet: {sheet} ---\n{df.to_string(index=False)}")

            docs = [Document(
                page_content="\n\n".join(all_text),
                metadata={"page": 0, "sheets": excel.sheet_names}
            )]

        else:
            raise ValueError("Unsupported file type")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100
        )
        return splitter.split_documents(docs)

    finally:
        os.unlink(temp_path)


# ---------------- VECTOR DB ----------------
def get_vector_collection():
    embedding_fn = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="mxbai-embed-large",  # nomic-embed-text:latest
)

    client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return client.get_or_create_collection(
        name="rag_app",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(chunks: list[Document], filename: str):
    collection = get_vector_collection()

    docs, metas, ids = [], [], []
    for i, chunk in enumerate(chunks):
        docs.append(chunk.page_content)
        meta = chunk.metadata.copy()
        meta["original_filename"] = filename
        meta["chunk_id"] = i
        metas.append(meta)
        ids.append(f"{filename}_{i}")

    collection.upsert(documents=docs, metadatas=metas, ids=ids)
    st.success(f"✅ Indexed {filename}")


# ---------------- QUERY ----------------
def query_collection(prompt: str, n_results=10):
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)

    if min(results["distances"][0]) > 0.7:
        return None

    return results


def results_to_documents(results):
    docs = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=text, metadata=meta))
    return docs


# ---------------- RERANKER ----------------
def re_rank_cross_encoders(prompt: str, docs: list[Document], top_k=3):
    encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(prompt, doc.page_content) for doc in docs]
    scores = encoder.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return ranked


# ---------------- LLM ----------------
def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3.2:1b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"}
        ]
    )
    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]


# ---------------- UI HELPERS ----------------
def display_citations(ranked_docs):
    st.subheader("📄 Source Citations")
    for doc, score in ranked_docs:
        st.markdown(
            f"**Source:** `{doc.metadata.get('original_filename')}` | "
            f"**Page:** {doc.metadata.get('page', 'N/A')}"
        )
        st.markdown(f"> {doc.page_content}")
        st.markdown("---")


# ---------------- MAIN APP ----------------
if __name__ == "__main__":

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = set()

    with st.sidebar:
        st.set_page_config(page_title="Document Vault")
        st.subheader("📁 Document Vault")

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if st.button("🔄 Resync Vault") and uploaded_files:
            for file in uploaded_files:
                if file.name in st.session_state.indexed_files:
                    continue

                splits = process_document(file)
                add_to_vector_collection(splits, file.name)
                st.session_state.indexed_files.add(file.name)

    st.header("🗣️ Document Vault Q&A")
    question = st.text_area("Ask a question")

    if st.button("🔥 Ask") and question:
        results = query_collection(question)

        if results is None:
            st.warning("Answer not found in documents.")
        else:
            retrieved_docs = results_to_documents(results)
            ranked_docs = re_rank_cross_encoders(question, retrieved_docs)
            
            # Store in session state
            st.session_state.ranked_docs = ranked_docs
            st.session_state.question = question

    # Show citations if available
    if "ranked_docs" in st.session_state:
        display_citations(st.session_state.ranked_docs)
        
        # AI Answer button
        if st.button("🤖 Generate AI Answer"):
            context = "\n\n".join(doc.page_content for doc, _ in st.session_state.ranked_docs)
            
            st.subheader("🤖 Answer")
            st.write_stream(call_llm(context, st.session_state.question))