# LocalRag

By Eytan Weill and the B-Yond team

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

