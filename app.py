import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langdetect import detect
from googletrans import Translator
import google.generativeai as genai

GEMINI_API_KEY = 'AIzaSyCn_xwNyzhWXMnTL6LupladblpGhLr4vkA'  # Your Gemini key
CSV_FILENAME = 'fdataset.csv'
SIM_THRESHOLD = 0.78

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
translator = Translator()
genai.configure(api_key=GEMINI_API_KEY)

df = pd.read_csv(CSV_FILENAME).fillna("")
dataset_questions = df['Incident Description'].astype(str).tolist()
embeddings = model.encode(dataset_questions)


def semantic_search(query, threshold=SIM_THRESHOLD):
    q_emb = model.encode([query])[0]
    sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    idx = np.argmax(sims)
    return idx if sims[idx] >= threshold else None


def gemini_answer(question):
    g_model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"Answer the following question in clear markdown format with headings, bullet points, and bold text:\n\n{question}"
    response = g_model.generate_content(prompt)
    return response.text


def respond(query):
    global df, embeddings  # Declare globals at the top before usage

    q_lang = detect(query)
    idx = semantic_search(query)
    if idx is not None:
        answer = df.iloc[idx]['Prevention Measures']
        if detect(answer) != q_lang:
            answer = translator.translate(answer, dest=q_lang).text
        return answer
    else:
        g_ans = gemini_answer(query)
        if detect(g_ans) != q_lang:
            g_ans = translator.translate(g_ans, dest=q_lang).text
        new_row = {
            "ID": len(df) + 1,
            "Incident Description": query,
            "Type of Cybercrime": "",
            "Affected Party": "",
            "Impact on Education": "",
            "Prevention Measures": g_ans,
            "Legal Framework": "",
            "Region": "",
            "Difficulty Level": "",
            "Keywords": ""
        }
        pd.DataFrame([new_row]).to_csv(CSV_FILENAME, mode='a', header=False, index=False)
        df.loc[len(df)] = new_row
        embeddings = np.vstack([embeddings, model.encode([query])])
        return g_ans



@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"answer": "No query sent."})
    try:
        answer = respond(user_query)
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

