from flask import Flask, request, jsonify, render_template, redirect, url_for
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
import chardet
import os, datetime

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
mongo = client["langchain_chat"]

API_KEY = "AIzaSyDjuXuDB3_h31mUkll00ZRF7UdsNiYfi4o"  # <-- replace with your Gemini API key
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Home / Upload Page ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # --- Read file content safely ---
        docs = []
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            docs.append(text)
        else:
            raw = open(filepath, "rb").read()
            result = chardet.detect(raw)
            encoding = result["encoding"] if result["encoding"] else "utf-8"
            with open(filepath, "r", encoding=encoding, errors="ignore") as f:
                docs.append(f.read())

        # --- Split into chunks ---
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = splitter.create_documents(docs)

        # --- Build FAISS index ---
        vector_db = FAISS.from_documents(texts, embeddings)
        vs_path = f"vectorstores/{filename}"
        vector_db.save_local(vs_path)

        # --- Save metadata in Mongo ---
        mongo.documents.insert_one({
            "filename": filename,
            "vectorstore_path": vs_path,
            "uploaded_at": datetime.datetime.utcnow()
        })

        return redirect(url_for("chat", filename=filename))

    # GET request: list all docs
    docs = [d["filename"] for d in mongo.documents.find({})]
    return render_template("index.html", docs=docs)

# --- Chat Page ---
@app.route("/chat/<filename>", methods=["GET", "POST"])
def chat(filename):
    docs = [d["filename"] for d in mongo.documents.find({})]

    # Fetch chat history
    history_docs = mongo.chats.find({"doc_filename": filename}).sort("timestamp", 1)
    history = [{"q": h["question"], "a": h["answer"]} for h in history_docs]

    if request.method == "POST":
        question = request.form.get("question")

        # Load FAISS index safely with explicit trust
        doc = mongo.documents.find_one({"filename": filename})
        vector_db = FAISS.load_local(
            doc["vectorstore_path"],
            embeddings,
            allow_dangerous_deserialization=True  # ✅ required for new LangChain versions
        )

        # RetrievalQA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY),
            retriever=vector_db.as_retriever()
        )
        answer = qa.invoke(question)["result"]

        # Save Q&A
        mongo.chats.insert_one({
            "doc_filename": filename,
            "question": question,
            "answer": answer,
            "timestamp": datetime.datetime.utcnow()
        })

        return redirect(url_for("chat", filename=filename))

    return render_template("chat.html", active_doc=filename, docs=docs, history=history)

# --- Run App ---
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("vectorstores", exist_ok=True)
    app.run(debug=True)
