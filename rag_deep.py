import streamlit as st
import shutil
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import os
import ollama
from ollama import ResponseError

# Initialize session state
if 'doc_processed' not in st.session_state:
    st.session_state.doc_processed = False
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "deepseek-r1:1.5b"

PROMPT_TEMPLATE = """You are an expert document analyst. Answer the user's query based *only* on the provided context.

Instructions:
- Answer concisely and directly.
- Do not include your thought process.
- If the information is not in the context, state that clearly.

Query: {user_query}
Context: {document_context}
Answer:"""

# Initialize models - HuggingFace for embeddings, Ollama for LLM
@st.cache_resource
def get_embedding_model():
    """Load the HuggingFace embedding model and cache it."""
    try:
        # Use HuggingFace embeddings (faster and more reliable)
        from langchain_huggingface import HuggingFaceEmbeddings
        
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ HuggingFace embedding error: {str(e)}")
        # Simple mock embedding as last resort
        class SimpleEmbeddings:
            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 384
        
        st.sidebar.warning("⚠️ Using simple embeddings (limited functionality)")
        return SimpleEmbeddings()

EMBEDDING_MODEL = get_embedding_model()

# Use Ollama DeepSeek for LLM responses
@st.cache_resource
def get_llm(model_name="deepseek-r1:1.5b"):
    try:
        llm = OllamaLLM(
            model=model_name, 
            base_url="http://localhost:11434"
        )
        return llm
    except Exception as e:
        st.sidebar.error(f"❌ LLM setup error for {model_name}: {str(e)}")
        return None

@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_ollama_status():
    """Check the status of Ollama and get the list of local models."""
    try:
        models = ollama.list()['models']
        # The response from ollama.list() can have 'name' or 'model' as the key for the model name.
        # We check for both to be robust.
        return True, [name for m in models if (name := m.get('name') or m.get('model'))]
    except ResponseError:
        return False, []

def clear_database():
    """Clear the vector store from session state."""
    if 'vector_db' in st.session_state:
        st.session_state.vector_db = None
    if 'doc_processed' in st.session_state:
        st.session_state.doc_processed = False
    if 'messages' in st.session_state:
        st.session_state.messages = []

def load_pdf_from_upload(uploaded_file):
    """Load PDF documents directly from uploaded file without saving to disk"""
    import tempfile
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name
    
    try:
        # Load the document from temporary file
        document_loader = PDFPlumberLoader(tmp_file_path)
        documents = document_loader.load()
        return documents
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)

def chunk_documents(raw_documents):
    # Use smaller chunks for faster embedding generation
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced from 1000 for faster processing
        chunk_overlap=100,  # Reduced from 200
        add_start_index=True
    )
    chunks = text_processor.split_documents(raw_documents)
    
    return chunks

def index_documents(document_chunks):
    """Index documents and store in session state using in-memory Chroma."""
    import time
    
    try:
        start_time = time.time()
        # Create and store vector DB in session state (in-memory)
        st.session_state.vector_db = Chroma.from_documents(
            documents=document_chunks, 
            embedding=get_embedding_model()
        )
        end_time = time.time()
        st.info(f"⏱️ Embedding generation took {end_time - start_time:.1f} seconds")
        st.session_state.doc_processed = True
        return True
    except Exception as e:
        st.error(f"❌ Error creating embeddings: {str(e)}")
        st.session_state.doc_processed = False
        return False

def find_related_documents(query):
    """Enhanced search that uses the vector DB from session state and removes duplicates."""
    try:
        if st.session_state.vector_db:
            results = st.session_state.vector_db.similarity_search(query, k=5)

            # De-duplicate results based on page_content
            unique_docs = []
            seen_content = set()
            for doc in results:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            return unique_docs
        else:
            return []
            
    except Exception as e:
        st.warning(f"Vector search failed: {str(e)}")
        return []

def generate_answer(user_query, relevant_docs, model_name):
    """Generate a streaming answer using the LLM and context"""
    language_model = get_llm(model_name)
    if language_model is None:
        def error_stream():
            yield "❌ LLM model not available. Please check if Ollama is running and the selected model is installed."
        return error_stream()
    
    # Create a prompt using the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the context from the documents
    document_context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Create the final prompt
    final_prompt = prompt_template.format(
        user_query=user_query, 
        document_context=document_context
    )
    
    try:
        # Return the stream from the LLM
        return language_model.stream(final_prompt)
    except Exception as e:
        def error_stream():
            yield f"❌ Error generating answer: {str(e)}"
        return error_stream()

# --- Streamlit App ---
st.set_page_config(page_title="Doc Q&A", page_icon="💬")

st.title("💬 RAG-Based Document Q&A System")
st.markdown("Upload a PDF, select a model, and ask questions about your document.")

# --- Sidebar for model selection and status ---
with st.sidebar:
    st.header("⚙️ Configuration")

    if st.button("Start New Chat"):
        clear_database()
        st.rerun()

    with st.expander("🧠 Model Selection", expanded=True):
        st.session_state.llm_model = st.selectbox(
            "Choose a model",
            ("deepseek-r1:1.5b", "llama3.2:1b"),
            index=0 if st.session_state.llm_model == "deepseek-r1:1.5b" else 1,
            help="Select the language model for generating answers."
        )

    with st.expander("🛠️ System Status", expanded=True):
        ollama_running, local_models = get_ollama_status()
        if not ollama_running:
            st.error("Ollama is not running. Please start Ollama to use the LLM features.")
        else:
            st.success("Ollama is running.")
            if st.session_state.llm_model not in local_models:
                st.warning(
                    f"Model '{st.session_state.llm_model}' not found locally. "
                    f"Please pull it with:\n" 
                    f"`ollama pull {st.session_state.llm_model}`"
                )

        if not get_embedding_model():
            st.error("Embedding model failed to load.")
        else:
            st.success("Embedding model loaded successfully.")

# --- Main App Logic ---

# Show file uploader only if a document has not been processed
if not st.session_state.doc_processed:
    st.info("Please upload a PDF document to begin.")
    uploaded_file = st.file_uploader(
        "Upload your PDF document", 
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file:
        with st.spinner("Reading and processing your document... This may take a moment."):
            # Clear any previous state before processing a new file
            clear_database()

            # Load and chunk the document
            raw_docs = load_pdf_from_upload(uploaded_file)
            chunks = chunk_documents(raw_docs)

            # Index the document chunks
            if chunks:
                st.info(f"📄 Document split into {len(chunks)} chunks. Now creating embeddings...")
                if index_documents(chunks):
                    st.success("✅ Document processed and indexed successfully!")
                    st.rerun()
                else:
                    st.error("❌ Document processing failed. Please check the errors above.")

# Show chat interface if a document has been processed
if st.session_state.doc_processed:
    st.header("💬 Chat with your document")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar='👤' if message["role"] == 'user' else '🤖'):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='👤'):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant", avatar='🤖'):
            with st.spinner("Thinking..."):
                relevant_docs = find_related_documents(prompt)
                
                if not relevant_docs:
                    response = "I couldn't find relevant information in the document to answer your question."
                    st.warning(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Display source documents for transparency
                    with st.expander("🔍 View Retrieved Context"):
                        for i, doc in enumerate(relevant_docs):
                            st.info(f"**Source {i+1}:**\n{doc.page_content}")
                    
                    # Use st.write_stream to display the streaming response
                    answer_stream = generate_answer(prompt, relevant_docs, st.session_state.llm_model)
                    full_answer = st.write_stream(answer_stream)
                    
                    # Add the complete assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_answer})
