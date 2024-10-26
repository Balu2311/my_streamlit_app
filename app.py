import streamlit as st
import openai
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
#import secrets

# Assuming constant.py is in the same directory and contains your OpenAI key
import constant

# Set your OpenAI API key
#openai.api_key = constant.open_ai_key
#openai.api_key = st.secrets["open_ai_key"]
openai.api_key = st.secrets["open_ai_key"]
print(openai.api_key)

# Set up logging
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

# Streamlit app
st.title("Conversational AI Chat")

# User input
user_query = st.text_input("Please describe your issue:")

if st.button("Submit"):
    if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
        st.error("Query must be a non-empty string")
    else:
        embeddings, texts, filenames = load_model('model.pkl')
        relevant_section, filename, similarity = retrieve_relevant_document(user_query, embeddings, texts, filenames)

        st.subheader("Response:")
        st.write(relevant_section)

        st.subheader("Document:")
        st.write(filename)

        st.subheader("Similarity Score:")
        st.write(similarity)
