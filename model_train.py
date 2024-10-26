import openai
import PyPDF2
import numpy as np
import joblib
import os
from docx import Document
import re
import constant

# Set your OpenAI API key
openai.api_key = constant.open_ai_key  # Replace with your actual API key

# Step 1: Extract text from a PDF document
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ''
            text += page_text
    return text.strip()  # Return trimmed text

# Load resources from DOCX files
def load_resources():
    resources = {}
    url_pattern = re.compile(r'https?://\S+')

    for root, dirs, files in os.walk('Sample Training Documents'):
        for file in files:
            if file.endswith('.docx'):
                doc = Document(os.path.join(root, file))
                description = ''
                referral_links = []
                
                for para in doc.paragraphs:
                    text = para.text.strip()
                    if text:
                        description += text + '\n'
                        # Find all URLs in the paragraph
                        found_links = url_pattern.findall(text)
                        referral_links.extend(found_links)

                # If no valid links were found, create a link based on the file name
                if not referral_links:
                    referral_links.append(f"https://vbnreddy/resources/{file}")

                resources[file] = {
                    "description": description.strip(),
                    "links": referral_links
                }
    
    return resources

# Step 2: Generate embedding for the text
def generate_embedding(text):
    if not text:
        return None  # Avoid generating embedding for empty text
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=text
    )
    return response['data'][0]['embedding']

# Step 3: Save embeddings and texts using joblib
def save_model(embeddings, texts, filenames, filename='model.pkl'):
    model = {'embeddings': embeddings, 'texts': texts, 'filenames': filenames}
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Step 4: Process multiple PDFs and store embeddings and text in a file
def process_and_save_pdfs(pdf_paths, model_filename='model.pkl'):
    embeddings = []
    texts = []
    filenames = []
    
    # Process PDF files
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        if text:  # Check if text is not empty
            embedding = generate_embedding(text)
            if embedding is not None:  # Ensure embedding is valid
                embeddings.append(embedding)
                texts.append(text)
                filenames.append(os.path.basename(pdf_path))
    
    # Load resources from DOCX files and generate embeddings
    resources = load_resources()
    for filename, resource in resources.items():
        embedding = generate_embedding(resource['description'])
        if embedding is not None:  # Ensure embedding is valid
            embeddings.append(embedding)
            texts.append(resource['description'])
            filenames.append(filename)
    
    save_model(embeddings, texts, filenames, model_filename)

# Example usage: Process multiple PDFs and save their embeddings in a file
pdf_files = [
    r"E://AA_values//dinesh//files//Air Compact Station Setup.pdf",
    r"E://AA_values//dinesh//files//Tamp Cylinder Not Lowering.pdf"
]
process_and_save_pdfs(pdf_files, 'model.pkl')
