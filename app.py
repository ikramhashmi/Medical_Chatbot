from flask import Flask, render_template, request, jsonify
from src.helper import huggingfaceembedding
from langchain_community.vectorstores import FAISS
from src.prompt import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from store_index import Database
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from src.prompt import *
import os

print("Starting the app...")
database = Database()
app = Flask(__name__)
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
embeddings = huggingfaceembedding()

print("Modeling....")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API_KEY,
    max_tokens=512
)

def truncate_text(text, max_length=500):
    """Ensure input text does not exceed model limits."""
    return text[:max_length] if len(text) > max_length else text

print("Resulting...")
results = database.as_retriever(search_kwargs={"k": 4})
q_chain = create_stuff_documents_chain(llm, prompts)
a_chain = create_retrieval_chain(results, q_chain)

print("ok")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")  # Use .get() to avoid KeyError
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    print("User Input:", msg)
    
    answers = a_chain.invoke({"input": msg})
    
    if "answer" in answers:
        response_text = answers["answer"]
    else:
        response_text = "Sorry, I couldn't process your request."

    print("Bot Response:", response_text)
    
    return jsonify({"response": response_text})  # Return JSON response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
