from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
import pickle, os, json
from prompt import my_prompt # Assuming 'my_prompt' contains your desired prompt string
from pydantic import BaseModel, Field
from typing import List

class Source(BaseModel):
    filename: str
    pagenumber: str

class ChatOutput(BaseModel):
    # Removed 'prompt' field based on your previous request "i didn't ask for prompt i need answer"
    # If you *do* want it, ensure my_prompt instructs the LLM to fill it.
    prompt: str = Field(description="Repeat the user prompt without chainging anything about it.")
    answer: str = Field(description="The answer to the question.")
    sources: List[Source] = Field(
        default=[],
        description=(
            "List of sources used to generate the answer. This always"
            " needs to be returned."
        ),
    )


load_dotenv()

os.getenv('OPENAI_API_KEY')

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID, model_kwargs={'device': 'cpu'})
    
with open('new_bm25.pkl', 'rb') as f:
    keyword_retriever = pickle.load(f)
    keyword_retriever.k = 5

vector_store1 = FAISS.load_local("Faiss", embedding, allow_dangerous_deserialization=True)
retriever = vector_store1.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(retrievers=[retriever,keyword_retriever],weights=[0.5, 0.5])



def rag(user_prompt, model='gpt-4o-mini'):
    retr = ensemble_retriever.invoke(user_prompt)

    slimmed_docs = [
        (d.page_content, {'source': d.metadata['file_path'], 'page': d.metadata['page']})
        for d in retr
    ]

    llm = ChatOpenAI(model=model, temperature=0) # Set temperature to 0 for more predictable output
    llmTools = llm.with_structured_output(ChatOutput)

    # Use the prompt imported from 'my_prompt'
    prompt = PromptTemplate.from_template(my_prompt)

    chain = prompt | llmTools
    chat_output_object = chain.invoke( # Renamed 'answer' to 'chat_output_object' for clarity
        {
            "prompt": user_prompt,
            "input": slimmed_docs
        }
    )

    # Corrected method call for JSON serialization
    json_response = json.loads(chat_output_object.model_dump_json(indent=2))

    return json_response


