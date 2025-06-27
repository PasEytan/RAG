# LocalRag

By Eytan Weill and the [B-Yond team](https://www.b-yond.com/)

## About This Project

This project is a locally-run Retrieval-Augmented Generation (RAG) system. It allows you to chat with your documents and get answers to your questions based on the information they contain, all without sending your data to external services.

## What is a Local RAG?

A standard RAG system often relies on third-party APIs (like for embedding and hosting your RAG) but a local RAG will do all of that on your own machine. This adds a level of flexibility and lowering constraints at the expense of being limited by the capabilities of your own machine. 

### Key Concepts in this RAG System:

* Chunking (using Langchain): To make it possible for the model to find relevant information in your documents, we first break down the text into smaller, manageable pieces called "chunks." Instead of feeding a whole book to the system at once, we provide it with smaller sections, which improves the accuracy of the retrieval process.

* Embedding (Using FAISS): Each chunk of text is then converted into a numerical representation, known as an "embedding" or a "vector." This process captures the semantic meaning of the text. When you ask a question, your query is also converted into an embedding. The system can then find the most relevant chunks from your documents by comparing the similarity of their embeddings to your question's embedding.

* Hybrid Search: Hybrid search elevates the retrieval process by combining two powerful search techniques: keyword-based search and semantic search.

    * Keyword Search (using BM25) is excellent at finding documents that contain the exact words or phrases from your query. It's precise and effective for matching specific terms.

    * Semantic Search, which uses the embeddings we've already discussed, finds documents that are contextually related in meaning, even if they don't use the same keywords.

## Quickstart

First, clone the repository and enter it: 

```
git clone https://github.com/PasEytan/RAG
cd RAG
```

Setup an .env file with an OpenAI key:

1. Create a file called ```.env```

2. Write at the very begining ```OPEN_AI_KEY=``` and then insert your OpenAI key.

> [!NOTE]
> If you are planning on using another LLM than gpt-4o-mini or an OpenAI model, be sure to change that in the rag.py file to whichever options you want. 

> [!NOTE]
> Be sure that all the python packages required for the RAG before running. This unfortunately has to be done by running the program and seeing what packages are missing (at least for now). 

### CLI:

Put all the PDFs you want to be part of the RAG into the ```pdfs``` folder.

Run the command:

```
python main.py
```

And voila! You are now running your own local RAG system.

### UI:

Make sure you have the streamlit python packaged installed with: 

```
pip install streamlit
```

and then run the command: 

```
streamlit run ui.py
```

You should now see a browser link appear in your terminal, click it and then you should be sent to a new tab in your browser. 

When that is done, insert each PDF file you want to embed or which folder full of PDFs you to select.

Click embed and once that is done, start using your RAG with a chatbot like UI. 

## Features

* Embedding your files can take a long time, but know that this is a one time process, once you have embedded your files, they will be ready always on startup. You will only have to embed again if you add or modify the files that are part of your RAG. Detecting if you have new files that are part of the RAG is done automatically.

* In the ```test.ipynb``` notebook, there is an experimental version with a RAG that chunks files by header instead of by a preset amount of characters. The program still has bugs and issues so your manual intervention is needed. 

## Testing

To test the accuracy of the RAG or any modifications done to it is easy.

First, go through the ```eval.ipynb``` file which will help you generate questions based on the PDFs you have given it. By the end of that process you should have a file called ```questions.csv```. 

When that is done, go to the notebook ```testing.ibynb``` This is a simple test that will use the RAG from the ```rag.py``` file and see how accurate the rag is at answering the questions in ```questions.csv```. 

> [!NOTE]
> From my testing, the RAG is about 90% accurate when it comes to response citations. I recommend tweaking the prompt used which is found in the ```prompt.py``` file or just creating a new one that is to your liking. Just remember to modify which prompt is being imported in the ```rag.py``` file. 

## Special Thanks 

I would like to give a special thanks to the [B-Yond team](https://www.b-yond.com/) for helping me develop this project and for giving me the resources to develop such a tool. 