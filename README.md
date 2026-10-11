# Multi-PDF RAG Assistant

<<<<<<< HEAD
A Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, LangChain, FAISS, and OpenAI. Upload multiple PDF documents, ask questions across them, and receive answers grounded in retrieved document excerpts with source filenames and page numbers.
=======
A Retrieval-Augmented Generation (RAG) application built with Python, Streamlit, LangChain, FAISS, OpenAI chat, and a local Hugging Face embedding model. Upload multiple PDF documents, ask questions across them, and receive answers grounded in retrieved document excerpts with source filenames and page numbers.
>>>>>>> 3002076 (git push origin main)

## Features

- Upload multiple PDF files at once.
- Extract PDF text and split it into manageable chunks.
<<<<<<< HEAD
- Generate embeddings using OpenAI `text-embedding-3-small`.
=======
- Generate embeddings locally using Hugging Face `sentence-transformers/all-MiniLM-L6-v2`.
>>>>>>> 3002076 (git push origin main)
- Index all uploaded PDFs in one FAISS vector database.
- Ask questions across the indexed documents using `gpt-4o-mini`.
- Display retrieved source excerpts with PDF filenames and page numbers.
- Run a cross-document review to flag possible disagreements in retrieved evidence.
- Display answer-generation latency and token usage.
- Optionally trace runs with LangSmith.
- Keep the FAISS index in Streamlit session state while the session is active.

## Technology Stack

- Python
- Streamlit
- LangChain
- OpenAI API
- FAISS
- PyPDF
- LangSmith (optional)

## Project Structure

```text
multi-rag/
├── rag_app.py
├── requirements.txt
<<<<<<< HEAD
├── .env                 # Created locally; do not commit
=======
├── .env                 # Create locally; do not commit
>>>>>>> 3002076 (git push origin main)
└── README.md
```

## Prerequisites

- Python 3.10 or a compatible Python version supported by the installed dependencies
- An OpenAI API key
- A LangSmith API key only if you want tracing

> OpenAI API usage may incur charges. A ChatGPT subscription does not include OpenAI API credits.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd multi-rag
```

Alternatively, create a project folder and place `rag_app.py` and `requirements.txt` inside it.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file in the project root, next to `rag_app.py`.

```dotenv
OPENAI_API_KEY=your_openai_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=multi-pdf-rag
```

Replace the placeholder values with your own keys.

If you do not want LangSmith tracing, you can use only the `OPENAI_API_KEY` entry and omit the LangSmith variables.

**Security:** Never commit `.env` or publish API keys. Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.py[cod]
```

## Run the Application

From the project directory, run:

```bash
streamlit run rag_app.py
```

Streamlit will provide a local URL to open in your browser.

## How It Works

1. **Upload PDFs:** Select multiple PDF documents in the Streamlit interface.
2. **Extract text:** `PyPDFLoader` extracts page content and metadata.
3. **Chunk documents:** `RecursiveCharacterTextSplitter` splits content into chunks using a chunk size of 1,000 characters and an overlap of 150 characters.
<<<<<<< HEAD
4. **Create embeddings:** OpenAI `text-embedding-3-small` converts chunks into vectors.
=======
4. **Create embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2` converts chunks into vectors locally.
>>>>>>> 3002076 (git push origin main)
5. **Build the vector store:** FAISS indexes all chunks from the uploaded PDFs in one searchable store.
6. **Retrieve evidence:** The retriever returns the six most similar chunks for a question.
7. **Generate an answer:** OpenAI `gpt-4o-mini` answers using the retrieved excerpts and is instructed to cite source identifiers.
8. **Show sources:** The interface lists the retrieved PDF filename, page number, and excerpt.
9. **Review possible conflicts:** A second model call reviews the retrieved excerpts for potential disagreements.

## Testing Suggestions

- Upload two or more PDFs in one operation.
- Ask a question whose answer is found in one PDF.
- Ask a question that requires evidence from multiple PDFs.
- Verify that each displayed source points to the expected filename and page.
- If possible, use test PDFs that contain conflicting statements and check the cross-document review.
- Test a question whose answer is not present in the uploaded documents.

## Current Limitations

- The FAISS index is kept in Streamlit session state; it is not persisted to disk by this version.
- The app indexes text extracted from PDFs. Scanned PDFs may require OCR before their content can be retrieved reliably.
- Source panels show retrieved excerpts; they do not guarantee that every displayed excerpt was explicitly cited in the generated answer.
- Conflict review examines the retrieved excerpts only. It may miss contradictions elsewhere in the PDFs, so verify important findings against the original documents.
- Token metrics are shown for answer generation; the separate conflict-review call also consumes tokens and is not included in those displayed figures.
- Page numbers are displayed as one-based page numbers.

## Dependencies

Install all dependencies with:

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
The requirements file includes Streamlit, LangChain, the OpenAI integration, community integrations, text splitters, PyPDF, FAISS, and python-dotenv.
=======
The requirements file includes Streamlit, LangChain, the OpenAI chat integration, Hugging Face/Sentence Transformers, community integrations, text splitters, PyPDF, FAISS, and python-dotenv.
>>>>>>> 3002076 (git push origin main)

## Assignment Deliverables

- `rag_app.py` — application source code
- `requirements.txt` — Python dependencies
- A screenshot showing an answer and its sources
- A screen recording demonstrating multi-PDF upload and question answering

## License

Add a license if you intend to distribute this project publicly.
