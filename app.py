from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from datetime import datetime

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chatModel = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11434")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ── In-memory chat history (server-side, no cookie size limit) ──
# Stored as a plain Python list — works perfectly for a single-user local app.
chat_history = []


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})
    answer = str(response["answer"])
    print("Bot:", answer)

    # Append to in-memory history
    chat_history.append({
        "question": msg,
        "answer": answer,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    return answer


@app.route("/history", methods=["GET"])
def get_history():
    """Return the full in-memory chat history as JSON."""
    return jsonify(chat_history)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Wipe the in-memory chat history."""
    chat_history.clear()
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)