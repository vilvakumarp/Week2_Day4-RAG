#This is SE-AI-B12-1-W2-D4 - RAG Assignment generated with Chatgpt
import os
import time
import hashlib
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "multi-pdf-rag")

# LangSmith environment configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = LANGSMITH_TRACING
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT


# =========================================================
# 2. APPLICATION CONFIGURATION
# =========================================================

APP_TITLE = "Multi-PDF RAG Assistant"

# Local Hugging Face embedding model. The model is downloaded once by
# sentence-transformers and then reused from the local Hugging Face cache.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 6

# Persist the FAISS index beside this Python script.
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
INDEX_METADATA_PATH = INDEX_DIR / "metadata.json"


# =========================================================
# 3. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-PDF RAG Assistant")
st.write(
    "Upload multiple PDF documents and ask questions "
    "across all of them. Every answer includes its sources."
)


# =========================================================
# 4. SESSION STATE
# =========================================================

if "store" not in st.session_state:
    st.session_state.store = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None

# Restore a previously saved local index when the app starts.
if st.session_state.store is None and INDEX_DIR.exists():
    try:
        saved_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        st.session_state.store = FAISS.load_local(
            str(INDEX_DIR),
            saved_embeddings,
            allow_dangerous_deserialization=True,
        )

        if INDEX_METADATA_PATH.exists():
            saved_metadata = json.loads(
                INDEX_METADATA_PATH.read_text(encoding="utf-8")
            )
            st.session_state.indexed_files = saved_metadata.get(
                "indexed_files", []
            )
            st.session_state.chunk_count = saved_metadata.get(
                "chunk_count", 0
            )
            st.session_state.file_signature = tuple(
                tuple(item) for item in saved_metadata.get(
                    "file_signature", []
                )
            ) or None
    except Exception as exc:
        st.session_state.store = None
        st.warning(f"Saved local FAISS index could not be loaded: {exc}")


# =========================================================
# 5. PDF UPLOAD
# =========================================================

st.header("Step 1: Upload PDF documents")

uploaded_files = st.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select several PDFs in one operation.",
)

if uploaded_files:
    st.write("### Selected files")

    for pdf_file in uploaded_files:
        st.write(
            f"📄 {pdf_file.name} "
            f"({pdf_file.size:,} bytes)"
        )

    # Create a fingerprint of the selected files.
    file_signature = tuple(
        sorted(
            (
                pdf_file.name,
                hashlib.sha256(pdf_file.getvalue()).hexdigest(),
            )
            for pdf_file in uploaded_files
        )
    )

    if (
        st.session_state.file_signature is not None
        and file_signature != st.session_state.file_signature
    ):
        st.warning(
            "The selected files differ from the indexed files. "
            "Rebuild the vector database to use the new selection."
        )

    if st.button(
        "Build / Rebuild Vector Database",
        type="primary",
    ):
        if not OPENAI_API_KEY:
            st.error(
                "OPENAI_API_KEY is missing. "
                "Please configure it in your .env file."
            )
            st.stop()

        all_chunks = []
        indexed_names = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )

        progress = st.progress(
            0,
            text="Preparing documents...",
        )

        try:
            # ---------------------------------------------
            # Process every uploaded PDF
            # ---------------------------------------------
            for index, pdf_file in enumerate(uploaded_files):

                st.write(f"Processing: {pdf_file.name}")

                # PyPDFLoader requires a file path.
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = Path(temp_dir) / pdf_file.name

                    pdf_path.write_bytes(
                        pdf_file.getvalue()
                    )

                    loader = PyPDFLoader(str(pdf_path))
                    pages = loader.load()

                    # Preserve the original filename.
                    for page in pages:
                        page.metadata["source"] = pdf_file.name

                    # Split pages into smaller chunks.
                    chunks = splitter.split_documents(pages)

                    all_chunks.extend(chunks)

                indexed_names.append(pdf_file.name)

                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Processed {pdf_file.name}",
                )

            if not all_chunks:
                st.error(
                    "No text was extracted from the PDFs. "
                    "Check that the documents contain readable text."
                )
                st.stop()

            # ---------------------------------------------
            # Create embeddings
            # ---------------------------------------------
            progress.progress(
                0.9,
                text="Generating embeddings and building FAISS index...",
            )

            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                encode_kwargs={"normalize_embeddings": True},
            )

            # All PDFs go into ONE vector database.
            new_store = FAISS.from_documents(
                documents=all_chunks,
                embedding=embeddings,
            )

            # Persist the index and its metadata beside this script.
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            new_store.save_local(str(INDEX_DIR))
            INDEX_METADATA_PATH.write_text(
                json.dumps(
                    {
                        "indexed_files": indexed_names,
                        "chunk_count": len(all_chunks),
                        "file_signature": file_signature,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            # Update state only after indexing succeeds.
            st.session_state.store = new_store
            st.session_state.indexed_files = indexed_names
            st.session_state.chunk_count = len(all_chunks)
            st.session_state.file_signature = file_signature

            progress.progress(
                1.0,
                text="Indexing completed!",
            )

            st.success(
                f"Successfully indexed {len(all_chunks)} chunks "
                f"from {len(indexed_names)} PDF document(s)."
            )

        except Exception as exc:
            st.error(f"Indexing failed: {exc}")


# =========================================================
# 6. DISPLAY DATABASE STATUS
# =========================================================

if st.session_state.store is not None:
    st.divider()
    st.header("Step 2: Vector Database Status")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Indexed PDFs",
        len(st.session_state.indexed_files),
    )

    col2.metric(
        "Total chunks",
        st.session_state.chunk_count,
    )

    col3.metric(
        "Embedding model",
        EMBEDDING_MODEL,
    )

    with st.expander("View indexed document names"):
        for filename in st.session_state.indexed_files:
            st.write(f"📄 {filename}")


# =========================================================
# 7. QUESTION-ANSWERING PROMPT
# =========================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a document question-answering assistant.

Answer the user's question using only the supplied
document excerpts.

RULES:

1. Do not invent facts that are not supported by the excerpts.
2. Cite supporting evidence using the source identifiers,
   such as [Source 1] or [Source 2].
3. If the documents do not contain the answer, say so clearly.
4. If two documents provide materially conflicting answers,
   explain the disagreement and identify both sources.
5. Do not assume that documents agree merely because no
   disagreement is immediately apparent.
6. Distinguish explicit statements from your own inference.
7. Keep the answer clear and concise.

DOCUMENT EXCERPTS:

{context}
""",
        ),
        (
            "human",
            "Question: {question}",
        ),
    ]
)


# =========================================================
# 8. DISAGREEMENT REVIEW PROMPT
# =========================================================

conflict_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Review the supplied document excerpts for disagreements
that are relevant to the user's question.

Return one of these classifications:

CONFLICT:
The excerpts contain materially incompatible claims.

NO CLEAR CONFLICT:
No material contradiction is apparent in the retrieved excerpts.

INSUFFICIENT EVIDENCE:
There is not enough comparable evidence to determine whether
the documents agree.

When a conflict exists:
- Explain the competing claims.
- Identify the source identifiers for both claims.
- Do not silently choose one document as correct.
- Do not invent missing evidence.

A missing statement in one excerpt does not automatically
mean that the documents contradict each other.
""",
        ),
        (
            "human",
            "Question: {question}\n\nDocument excerpts:\n{context}",
        ),
    ]
)


# =========================================================
# 9. ASK QUESTIONS
# =========================================================

st.divider()
st.header("Step 3: Ask Questions")

question = st.text_input(
    "Enter your question",
    placeholder="Example: How many leave days do new joiners get?",
)

if st.button("Get Answer", type="primary"):

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is missing.")

    elif st.session_state.store is None:
        st.warning(
            "Please upload PDFs and build the vector database first."
        )

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            llm = ChatOpenAI(
                model=CHAT_MODEL,
                temperature=0,
                api_key=OPENAI_API_KEY,
            )

            # ---------------------------------------------
            # Retrieve relevant document chunks
            # ---------------------------------------------
            retriever = st.session_state.store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K},
            )

            with st.spinner("Searching the documents..."):
                retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                st.warning("No relevant document chunks were found.")
                st.stop()

            # ---------------------------------------------
            # Prepare source-labelled context
            # ---------------------------------------------
            context_parts = []
            source_lookup = {}

            for index, doc in enumerate(retrieved_docs, start=1):
                filename = Path(
                    doc.metadata.get("source", "Unknown")
                ).name

                # PyPDFLoader uses zero-based page numbers.
                page_number = int(
                    doc.metadata.get("page", 0)
                ) + 1

                source_id = f"Source {index}"

                context_parts.append(
                    f"[{source_id}: {filename}, page {page_number}]\n"
                    f"{doc.page_content}"
                )

                source_lookup[source_id] = {
                    "filename": filename,
                    "page": page_number,
                    "content": doc.page_content,
                }

            context = "\n\n".join(context_parts)

            # ---------------------------------------------
            # Generate the answer and measure latency
            # ---------------------------------------------
            with st.spinner("Generating an answer..."):
                start_time = time.perf_counter()

                answer_chain = answer_prompt | llm

                answer_response = answer_chain.invoke(
                    {
                        "context": context,
                        "question": question,
                    },
                    config={
                        "run_name": "Multi-PDF RAG Answer",
                        "tags": ["multi-pdf-rag", "answer"],
                    },
                )

                answer_latency = (
                    time.perf_counter() - start_time
                )

            # ---------------------------------------------
            # Display the answer
            # ---------------------------------------------
            st.subheader("Answer")
            st.markdown(answer_response.content)

            # ---------------------------------------------
            # Display cited source chunks
            # ---------------------------------------------
            st.subheader("Retrieved Sources")

            # Deduplicate repeated filename/page combinations.
            unique_sources = {}

            for source_id, source in source_lookup.items():
                key = (
                    source["filename"],
                    source["page"],
                )

                if key not in unique_sources:
                    unique_sources[key] = {
                        "content": source["content"],
                        "source_ids": [source_id],
                    }
                else:
                    unique_sources[key]["source_ids"].append(
                        source_id
                    )

            for (filename, page_number), source in unique_sources.items():
                with st.expander(
                    f"📄 {filename} — Page {page_number}"
                ):
                    st.caption(
                        "Retrieved as: "
                        + ", ".join(source["source_ids"])
                    )
                    st.write(source["content"])

            # ---------------------------------------------
            # Check for possible disagreements
            # ---------------------------------------------
            st.subheader("Cross-Document Review")

            with st.spinner(
                "Checking retrieved evidence for disagreements..."
            ):
                conflict_chain = conflict_prompt | llm

                conflict_response = conflict_chain.invoke(
                    {
                        "question": question,
                        "context": context,
                    },
                    config={
                        "run_name": "Cross-Document Conflict Review",
                        "tags": ["multi-pdf-rag", "conflict-review"],
                    },
                )

            st.markdown(conflict_response.content)

            # ---------------------------------------------
            # Latency and token usage
            # ---------------------------------------------
            st.subheader("Performance Information")

            usage = (
                answer_response.usage_metadata or {}
            )

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get(
                "total_tokens",
                input_tokens + output_tokens,
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Answer latency",
                f"{answer_latency:.2f} sec",
            )

            metric2.metric(
                "Input tokens",
                input_tokens,
            )

            metric3.metric(
                "Output tokens",
                output_tokens,
            )

            st.caption(
                f"Total answer tokens: {total_tokens}. "
                "Conflict-review usage is not included in these figures."
            )

            if LANGSMITH_TRACING == "true":
                st.caption(
                    f"LangSmith project: {LANGSMITH_PROJECT}"
                )

        except Exception as exc:
            st.error(
                f"Question answering failed: {exc}"
            )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    "Multi-PDF RAG Assistant | "
    "Python + Streamlit + LangChain + FAISS + OpenAI"
)
