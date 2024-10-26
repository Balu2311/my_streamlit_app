from flask import Flask, request, jsonify, render_template
import openai
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from model_train import process_and_save_pdfs
import os
import logging

app = Flask(__name__)
import constant

# Set your OpenAI API key
openai.api_key = constant.open_ai_key
#openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)

def load_model(filename='model.pkl'):
    try:
        model = joblib.load(filename)
        return model['embeddings'], model['texts'], model['filenames']
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return [], [], []

def generate_query_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=query
    )
    return response['data'][0]['embedding']

def find_most_similar(query_embedding, stored_embeddings):
    similarities = cosine_similarity([query_embedding], stored_embeddings)
    return np.argmax(similarities), np.max(similarities)

def retrieve_relevant_document(query, embeddings, texts, filenames):
    query_embedding = generate_query_embedding(query)
    index, similarity = find_most_similar(query_embedding, embeddings)
    return texts[index], filenames[index], similarity

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get("query")

    if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
        return jsonify({"error": "Query must be a non-empty string"}), 400

    embeddings, texts, filenames = load_model('model.pkl')
    relevant_section, filename, similarity = retrieve_relevant_document(user_query, embeddings, texts, filenames)

    return render_template('query.html', response=relevant_section, document=filename, similarity=similarity)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
