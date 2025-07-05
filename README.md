# RAG-Based Domain-Specific Q&A System

This repository contains a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Ollama. The application allows users to upload PDF documents and ask questions about their content, with support for multiple language models.


## Features

- **PDF Document Upload**: Upload PDF files directly through the web interface.
- **Multi-LLM Support**: Switch between different language models like **DeepSeek** and **LLaMA3** for generating answers.
- **Efficient Vector Search**: Uses `ChromaDB` for storing and retrieving document embeddings.
- **SentenceTransformer Embeddings**: Employs `HuggingFace` sentence-transformer models for accurate semantic search.
- **Optimized Retrieval**: Retrieves the top 5 most relevant, unique document chunks to provide rich context.
- **Interactive Chat Interface**: A persistent and user-friendly chat interface for conversational Q&A.
- **Performance-Tuned**: Implements caching to ensure fast model loading and a smooth user experience.
- **Transparent Context**: View the exact document chunks used to generate an answer.
- **Robust Error Handling**: Includes fallback mechanisms for model loading and document processing.

## How It Works

The application follows a RAG pipeline:

1.  **File Upload**: The user uploads a PDF file. The existing vector database is cleared to ensure responses are based only on the current document.
2.  **Document Loading and Chunking**: The PDF is loaded using `PDFPlumberLoader` and split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding and Indexing**: `HuggingFace` sentence-transformer embeddings are generated for each chunk, and they are stored in a `Chroma` vector store.
4.  **User Query**: The user asks a question through the chat interface and selects a language model.
5.  **Similarity Search**: The user's question is embedded, and a similarity search is performed on the `Chroma` vector store to find the top 5 most relevant document chunks.
6.  **LLM-Powered Answer Generation**: The retrieved chunks and the user's question are passed to the selected Ollama LLM (DeepSeek or LLaMA3) through an optimized prompt.
7.  **Display Answer**: The LLM-generated answer is streamed back to the user in the chat interface.

## Technologies Used

- **Streamlit**: For the web application and user interface.
- **LangChain**: For document loading, text splitting, and orchestrating the RAG pipeline.
- **Ollama**: For running the language models locally.
- **ChromaDB**: For the vector store.
- **HuggingFace Transformers**: For generating sentence embeddings.
- **PDFPlumber**: For extracting text from PDF documents.

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Gen-AI-With-Deep-Seek-R1
```

### 2. Install and Set Up Ollama

- Follow the instructions at [ollama.ai](https://ollama.ai/) to install and run Ollama on your system.
- Pull the required language models:

  ```bash
  ollama pull deepseek-r1:1.5b
  ollama pull llama3:8b
  ```

### 3. Install Python Dependencies

It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

Once the setup is complete, run the following command to start the application:

```bash
streamlit run rag_deep.py
```

The application will open in your default web browser.

## Usage

1.  **Upload a PDF**: Click the "Upload a PDF file" button to select and upload your document.
2.  **Select a Model**: Use the dropdown in the sidebar to choose between `deepseek-r1:1.5b` and `llama3:8b`.
3.  **Ask a Question**: Type your question in the chat input box and press Enter.
4.  **View Sources**: Expand the "View Retrieved Context" section to see the document chunks used to generate the answer.