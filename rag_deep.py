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
- If the information is not in the context, respond with EXACTLY: "I cannot answer this question based on the document content."
- Do NOT use your general knowledge to answer questions outside of the provided context.

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

def find_related_documents(query, similarity_threshold=0.25):
    """Enhanced search that uses the vector DB from session state and removes duplicates.
    
    Args:
        query: The user's query to search for
        similarity_threshold: Minimum similarity score (0-1) required to consider a document relevant
    """
    try:
        if st.session_state.vector_db:
            # Use similarity_search_with_score to get both documents and scores
            results = st.session_state.vector_db.similarity_search_with_score(query, k=5)

            # Filter results based on similarity threshold
            filtered_results = []
            seen_content = set()
            
            for doc, score in results:
                # Chromadb returns distance, not similarity, so convert (1.0 = exact match, higher numbers = less similar)
                # Most embedding distances are in the range of 0-2, where 0 is perfect match
                similarity = 1.0 - (score / 2.0)  # Convert to 0-1 range where 1 is perfect match
                
                if similarity >= similarity_threshold and doc.page_content not in seen_content:
                    filtered_results.append(doc)
                    seen_content.add(doc.page_content)
            
            return filtered_results
        else:
            return []
            
    except Exception as e:
        st.warning(f"Vector search failed: {str(e)}")
        return []

def generate_answer(user_query, relevant_docs, model_name):
    """Generate a streaming answer using the LLM and context
    
    Only documents meeting the similarity threshold will be in relevant_docs.
    If no documents meet the threshold, a standard "cannot answer" response is returned.
    """
    language_model = get_llm(model_name)
    if language_model is None:
        def error_stream():
            yield "❌ LLM model not available. Please check if Ollama is running and the selected model is installed."
        return error_stream()
    
    # If there are no relevant docs or the content is too limited, don't use the LLM
    if not relevant_docs:
        def not_found_stream():
            yield "I cannot answer this question based on the document content."
        return not_found_stream()
    
    # Create a prompt using the template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the context from the documents
    document_context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Check if the retrieved content is substantial enough (min 20 characters)
    if len(document_context.strip()) < 20:
        def insufficient_stream():
            yield "I cannot answer this question based on the document content."
        return insufficient_stream()
    
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
        
        # Add similarity threshold slider
        if 'similarity_threshold' not in st.session_state:
            st.session_state.similarity_threshold = 0.25
            
        st.session_state.similarity_threshold = st.slider(
            "Relevance threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="Minimum similarity score required to consider a document relevant. Higher values mean stricter matching."
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
                # Use the current similarity threshold from session state
                relevant_docs = find_related_documents(
                    prompt, 
                    similarity_threshold=st.session_state.similarity_threshold
                )
                
                # Display source documents for transparency if any were found
                if relevant_docs:
                    with st.expander("🔍 View Retrieved Context"):
                        for i, doc in enumerate(relevant_docs):
                            st.info(f"**Source {i+1}:**\n{doc.page_content}")
                
                # Always use generate_answer - it will now handle empty documents appropriately
                answer_stream = generate_answer(prompt, relevant_docs, st.session_state.llm_model)
                
                # If no documents were found, display a warning style for the message
                if not relevant_docs:
                    st.warning(f"No information found in the document that meets the relevance threshold ({st.session_state.similarity_threshold:.2f}). Try adjusting the threshold in the sidebar.")
                
                # Use st.write_stream to display the streaming response
                full_answer = st.write_stream(answer_stream)
                
                # Add the complete assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
