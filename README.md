# Project Description

This project is a **smart document chatbot** that allows users to ask questions about documents (like PDFs or text files) and get precise answers. It works by:

- **Splitting the document** into smaller chunks.  
- **Creating vector embeddings** for each chunk using a local model (`sentence-transformers`).  
- **Storing the embeddings** in a **FAISS vector database** for fast similarity search.  
- **Using Gemini Flash** to generate human-like answers based on the retrieved relevant chunks.

In short: **upload a document → ask a question → get an intelligent answer.**
