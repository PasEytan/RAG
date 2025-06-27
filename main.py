import pickle
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever


def find_pdfs(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    # The rglob pattern is case-sensitive by default; to catch .PDF too, you could check suffix lowercased:
    pdfs = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.pdf']
    return pdfs


def func():
    from rag import rag

    print("\nready to start\n")
    while True:
        user_prompt = str(input('Say something:(write "quit" to quit)\n'))

        if user_prompt.replace(" ", '') == "quit":
            break
        answer = rag(user_prompt)
        print(answer['answer'])
        print("\nCitations:")
        for i in answer['sources']:
            print(f'{i['filename']} on page {i['pagenumber']}')

    return

def embed(pdfs):
    print("Your filesystem has not been regognized, embedding new files...\n")
    
    t = []
    for i in pdfs:
        FILE_PATH = str(i)  

        doc = PyMuPDFLoader(FILE_PATH, mode='page') # open a document
        text = doc.load()

        for i in range(len(text)):
            text[i].metadata['page'] += 1
        t.extend(text)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(t)

    EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs={'device': 'cpu'})

    vector_store = FAISS.from_documents(texts, embedding)

    vector_store.save_local("Faiss")

    keyword_retriever = BM25Retriever.from_documents(t)


    with open('new_bm25.pkl', "wb") as f:
        pickle.dump(keyword_retriever, f)

    return func()


if __name__ == "__main__":
    print("Setting up environment, please wait...\n")


    file_path = "./pdfs/"

    pdfs = find_pdfs(file_path)

    if Path('./temp.txt').exists() == False or open('./temp.txt').read() != str(pdfs):
        with open('./temp.txt', 'w') as f:
            f.write(str(pdfs))
        embed(pdfs)        

    else:
        func()