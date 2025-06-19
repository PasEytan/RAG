import streamlit as st
import pickle
import os
import json
import time
from pathlib import Path
from typing import List
import tempfile
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

# Import your existing prompt
try:
    from prompt import my_prompt
except ImportError:
    # Fallback prompt if prompt.py is not available
    my_prompt = """
    You are a helpful assistant that answers questions based on the provided context.
    
    Context: {input}
    
    Question: {prompt}
    
    Please provide a comprehensive answer based on the context provided, and list the sources you used.
    """

# Pydantic models (same as your rag.py)
class Source(BaseModel):
    filename: str
    pagenumber: str

class ChatOutput(BaseModel):
    prompt: str = Field(description="Repeat the user prompt without changing anything about it.")
    answer: str = Field(description="The answer to the question.")
    sources: List[Source] = Field(
        default=[],
        description=(
            "List of sources used to generate the answer. This always"
            " needs to be returned."
        ),
    )

# Load environment variables
load_dotenv()

# Constants
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
UPLOAD_DIR = "./uploaded_pdfs"
FAISS_DIR = "./Faiss"
BM25_FILE = "./new_bm25.pkl"
TEMP_FILE = "./temp.txt"

# Initialize session state
if 'embedded_files' not in st.session_state:
    st.session_state.embedded_files = []
if 'retriever_loaded' not in st.session_state:
    st.session_state.retriever_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'embedding_in_progress' not in st.session_state:
    st.session_state.embedding_in_progress = False
if 'embeddings_ready' not in st.session_state:
    st.session_state.embeddings_ready = False

def find_pdfs(root_dir: str) -> List[Path]:
    """Find all PDF files in the directory (same as your original function)"""
    root = Path(root_dir)
    pdfs = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.pdf']
    return pdfs

def embed_documents(pdfs: List[Path]):
    """Embed documents using your existing logic"""
    st.session_state.embedding_in_progress = True
    st.session_state.embeddings_ready = False
    
    with st.spinner("Embedding documents... This may take a few minutes."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Load and process documents
            t = []
            status_text.text("Loading PDF documents...")
            for idx, pdf_path in enumerate(pdfs):
                FILE_PATH = str(pdf_path)
                doc = PyMuPDFLoader(FILE_PATH, mode='page')
                text = doc.load()
                
                # Adjust page numbers (same as your logic)
                for i in range(len(text)):
                    text[i].metadata['page'] += 1
                t.extend(text)
                
                progress_bar.progress((idx + 1) / (len(pdfs) * 3))
                status_text.text(f"Loading PDF {idx + 1}/{len(pdfs)}: {pdf_path.name}")
            
            # Text splitting
            status_text.text("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.split_documents(t)
            progress_bar.progress(len(pdfs) / (len(pdfs) * 3))
            
            # Create embeddings
            status_text.text("Creating embeddings... This may take several minutes.")
            embedding = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_ID, 
                model_kwargs={'device': 'cpu'}
            )
            
            # Create and save vector store
            vector_store = FAISS.from_documents(texts, embedding)
            vector_store.save_local(FAISS_DIR)
            progress_bar.progress((len(pdfs) * 2) / (len(pdfs) * 3))
            
            # Create and save BM25 retriever
            status_text.text("Creating BM25 retriever...")
            keyword_retriever = BM25Retriever.from_documents(t)
            with open(BM25_FILE, "wb") as f:
                pickle.dump(keyword_retriever, f)
            
            progress_bar.progress(1.0)
            status_text.text("Embedding complete!")
            
            # Update temp file to track embedded files
            with open(TEMP_FILE, 'w') as f:
                f.write(str(pdfs))
            
            st.session_state.embedding_in_progress = False
            st.session_state.embeddings_ready = True
            st.success(f"Successfully embedded {len(pdfs)} PDF files!")
            
        except Exception as e:
            st.session_state.embedding_in_progress = False
            st.session_state.embeddings_ready = False
            st.error(f"Error during embedding: {str(e)}")
            raise e

def load_retriever():
    """Load the retriever system"""
    if not st.session_state.retriever_loaded:
        try:
            # Load embedding model
            embedding = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_ID, 
                model_kwargs={'device': 'cpu'}
            )
            
            # Load BM25 retriever
            with open(BM25_FILE, 'rb') as f:
                keyword_retriever = pickle.load(f)
                keyword_retriever.k = 5
            
            # Load FAISS vector store
            vector_store = FAISS.load_local(
                FAISS_DIR, 
                embedding, 
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever, keyword_retriever],
                weights=[0.5, 0.5]
            )
            
            st.session_state.ensemble_retriever = ensemble_retriever
            st.session_state.retriever_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading retriever: {str(e)}")
            return False
    return True

def rag_query(user_prompt: str, model: str = 'gpt-4o-mini'):
    """Execute RAG query (same logic as your rag.py)"""
    if not st.session_state.retriever_loaded:
        if not load_retriever():
            return None
    
    # Retrieve relevant documents
    retr = st.session_state.ensemble_retriever.invoke(user_prompt)
    
    slimmed_docs = [
        (d.page_content, {'source': d.metadata['file_path'], 'page': d.metadata['page']})
        for d in retr
    ]
    
    # Create LLM and structured output
    llm = ChatOpenAI(model=model, temperature=0)
    llmTools = llm.with_structured_output(ChatOutput)
    
    # Use the prompt
    prompt = PromptTemplate.from_template(my_prompt)
    
    chain = prompt | llmTools
    chat_output_object = chain.invoke({
        "prompt": user_prompt,
        "input": slimmed_docs
    })
    
    # Convert to JSON
    json_response = json.loads(chat_output_object.model_dump_json(indent=2))
    return json_response

def main():
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Document Assistant")
    st.markdown("Upload PDF documents and ask questions about their content!")
    
    # Create upload directory if it doesn't exist
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Check for existing embedded files
        existing_pdfs = find_pdfs(UPLOAD_DIR)
        
        # Show currently embedded files
        if existing_pdfs:
            st.subheader("Current Documents:")
            for pdf in existing_pdfs:
                st.text(f"📄 {pdf.name}")
        
        # Tab selection for upload method
        upload_tab = st.radio(
            "Choose upload method:",
            ["Upload Files", "Upload Folder"],
            horizontal=True
        )
        
        if upload_tab == "Upload Files":
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload PDF Files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Select one or more PDF files to add to your knowledge base"
            )
            
            # Process uploaded files
            if uploaded_files:
                new_files = []
                existing_files = []
                
                for uploaded_file in uploaded_files:
                    file_path = Path(UPLOAD_DIR) / uploaded_file.name
                    
                    # Always save uploaded file (overwrite if exists)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    if file_path not in new_files:
                        new_files.append(file_path)
                
                if new_files:
                    st.success(f"Uploaded {len(new_files)} files: {[f.name for f in new_files]}")
                    
                    # Show embed button for uploaded files
                    if st.button("🔄 Embed Uploaded Documents", key="embed_files"):
                        with st.spinner("Starting embedding process..."):
                            st.session_state.embedding_in_progress = True
                            # Force a rerun to show the embedding progress
                            st.rerun()
        
        else:  # Upload Folder
            st.subheader("📁 Folder Upload")
            folder_path = st.text_input(
                "Enter folder path:",
                placeholder="/path/to/your/pdf/folder",
                help="Enter the full path to a folder containing PDF files"
            )
            
            if folder_path and st.button("📂 Load PDFs from Folder"):
                folder_path_obj = Path(folder_path)
                if folder_path_obj.exists() and folder_path_obj.is_dir():
                    folder_pdfs = find_pdfs(folder_path)
                    if folder_pdfs:
                        # Copy PDFs to upload directory
                        copied_files = []
                        for pdf_path in folder_pdfs:
                            dest_path = Path(UPLOAD_DIR) / pdf_path.name
                            shutil.copy2(pdf_path, dest_path)
                            copied_files.append(dest_path)
                        
                        st.success(f"Loaded {len(copied_files)} PDF files from folder: {[f.name for f in copied_files]}")
                        st.session_state.folder_files_loaded = copied_files
                    else:
                        st.warning("No PDF files found in the specified folder or its subfolders")
                else:
                    st.error("Invalid folder path or folder doesn't exist")
            
            # Show embed button for folder files
            if hasattr(st.session_state, 'folder_files_loaded') and st.session_state.folder_files_loaded:
                if st.button("🔄 Embed Folder Documents", key="embed_folder"):
                    with st.spinner("Starting embedding process..."):
                        st.session_state.embedding_in_progress = True
                        st.rerun()
        
        # Check if we need to embed existing files
        if existing_pdfs and not Path(FAISS_DIR).exists():
            st.warning("Found PDF files but no embeddings. Please embed documents first.")
            if st.button("🚀 Embed Existing Documents"):
                with st.spinner("Starting embedding process..."):
                    st.session_state.embedding_in_progress = True
                    st.rerun()
        
        # Status indicator
        if st.session_state.embedding_in_progress:
            st.error("🔄 EMBEDDING IN PROGRESS - Please wait...")
        elif Path(FAISS_DIR).exists() and Path(BM25_FILE).exists():
            st.success("✅ Embeddings ready - You can chat now!")
        else:
            st.info("📝 No embeddings found - Upload and embed documents first")
        
        # Clear all documents
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                # Remove all files and embeddings
                shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
                shutil.rmtree(FAISS_DIR, ignore_errors=True)
                if Path(BM25_FILE).exists():
                    Path(BM25_FILE).unlink()
                if Path(TEMP_FILE).exists():
                    Path(TEMP_FILE).unlink()
                
                # Reset session state
                st.session_state.embedded_files = []
                st.session_state.retriever_loaded = False
                st.session_state.chat_history = []
                st.session_state.confirm_clear = False
                
                Path(UPLOAD_DIR).mkdir(exist_ok=True)
                st.success("All documents cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm deletion of all documents")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Check embedding status and handle embedding process
    if st.session_state.embedding_in_progress:
        st.error("🔄 EMBEDDING IN PROGRESS")
        st.info("Documents are being processed and embedded. This may take several minutes depending on the number of documents.")
        
        # Actually perform the embedding
        all_pdfs = find_pdfs(UPLOAD_DIR)
        if all_pdfs:
            try:
                embed_documents(all_pdfs)
                st.session_state.retriever_loaded = False  # Force reload
                st.session_state.embedding_in_progress = False
                st.session_state.embeddings_ready = True
                st.success("🎉 Embedding completed successfully! You can now chat with your documents.")
                st.balloons()
                time.sleep(2)  # Give user time to see success message
                st.rerun()
            except Exception as e:
                st.session_state.embedding_in_progress = False
                st.error(f"❌ Embedding failed: {str(e)}")
        else:
            st.session_state.embedding_in_progress = False
            st.error("No PDF files found to embed!")
        return
    
    # Check if we have embedded documents
    if not Path(FAISS_DIR).exists() or not Path(BM25_FILE).exists():
        st.info("👆 Please upload and embed some PDF documents first using the sidebar.")
        return
    
    # Initialize embeddings ready status
    if not st.session_state.embeddings_ready and Path(FAISS_DIR).exists() and Path(BM25_FILE).exists():
        st.session_state.embeddings_ready = True
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know?",
            key="user_input"
        )
    
    with col2:
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Process query
    if (ask_button or user_input) and user_input.strip():
        with st.spinner("Searching through your documents..."):
            response = rag_query(user_input.strip())
            
            if response:
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_input.strip(),
                    'response': response
                })
                
                # Note: We can't clear the input due to Streamlit limitations
                # The input will remain until the user manually clears it or refreshes
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("📋 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}{'...' if len(chat['question']) > 100 else ''}", expanded=(i == 0)):
                st.markdown("**Question:**")
                st.write(chat['question'])
                
                st.markdown("**Answer:**")
                st.write(chat['response']['answer'])
                
                if chat['response']['sources']:
                    st.markdown("**Sources:**")
                    for source in chat['response']['sources']:
                        st.write(f"📄 {source['filename']} (Page {source['pagenumber']})")
                
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit and LangChain*")

if __name__ == "__main__":
    main()