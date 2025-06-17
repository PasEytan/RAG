import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
import pickle

def rag(user_prompt, TOP_K):
    user_prompt = user_prompt

    EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs={'device': 'cpu'})
        
    with open('test.pkl', 'rb') as f:
        keyword_retriever = pickle.load(f)
        keyword_retriever.k = 5
    
    vector_store1 = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    retriever = vector_store1.as_retriever(search_kwargs={"k": TOP_K})

    ensemble_retriever = EnsembleRetriever(retrievers=[retriever,keyword_retriever],weights=[0.5, 0.5])


    retr = ensemble_retriever.get_relevant_documents(user_prompt)

    texts= [(d.page_content, d.metadata) for d in retr]

    return answer
