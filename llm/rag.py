import logging
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from PyPDF2 import PdfReader
import torch


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models")
INDEX_PATH = "/models/faiss_index"
PDF_DIR = "/pdfs"

logger.info(f"Loading model from: {MODEL_PATH}")
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    logger.info("Model and tokenizer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

sentence_transformer = SentenceTransformer('BAAI/bge-m3')

SYSTEM_MESSAGE = """You are a helpful assistant. Answer the user's question based ONLY on the given context.
If the context doesn't contain relevant information to the specific question, say 'I don't have enough information to answer that specific question.'
Do not make up information or use general knowledge outside of the given context."""

def split_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdf(pdf_content):
    reader = PdfReader(pdf_content)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return split_text(text)

def vectorize_text(text):
    return sentence_transformer.encode([text], normalize_embeddings=True)[0]

def ensure_index_exists():
    if not os.path.exists(INDEX_PATH):
        logger.info("Faiss index not found. Creating a new one.")
        dimension = sentence_transformer.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        faiss.write_index(index, INDEX_PATH)
        with open(INDEX_PATH + '.metadata', 'wb') as f:
            pickle.dump([], f)
    else:
        logger.info("Faiss index found.")

def update_index(vectors, metadata):
    if not os.path.exists(INDEX_PATH):
        dimension = vectors[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        existing_metadata = []
    else:
        index = faiss.read_index(INDEX_PATH)
        with open(INDEX_PATH + '.metadata', 'rb') as f:
            existing_metadata = pickle.load(f)

    vectors_np = np.array(vectors).astype('float32')
    if vectors_np.shape[0] > 0:
        index.add(vectors_np)
        existing_metadata.extend(metadata)
        faiss.write_index(index, INDEX_PATH)
        with open(INDEX_PATH + '.metadata', 'wb') as f:
            pickle.dump(existing_metadata, f)

    logger.info(f"Index updated. Total vectors: {index.ntotal}, Dimension: {index.d}")

def process_pdf_directory():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {PDF_DIR}")
        return 0

    vectors = []
    metadata = []
    for filename in pdf_files:
        file_path = os.path.join(PDF_DIR, filename)
        with open(file_path, 'rb') as file:
            chunks = process_pdf(file)
            for i, chunk in enumerate(chunks):
                vector = vectorize_text(chunk)
                vectors.append(vector)
                metadata.append({
                    'filename': filename,
                    'chunk_id': i,
                    'text': chunk
                })

    if vectors:
        update_index(vectors, metadata)
    logger.info(f"Processed {len(pdf_files)} PDFs into {len(vectors)} chunks")
    return len(pdf_files)

def search_documents(query, index, metadata, k=5):
    query_vector = vectorize_text(query)
    D, I = index.search(query_vector.reshape(1, -1), k)
    results = []
    for i, (idx, distance) in enumerate(zip(I[0], D[0])):
        results.append((metadata[idx], float(distance)))
    return results

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.pdf'):
        chunks = process_pdf(file)
        vectors = []
        metadata = []
        for i, chunk in enumerate(chunks):
            vector = vectorize_text(chunk)
            vectors.append(vector)
            metadata.append({
                'filename': file.filename,
                'chunk_id': i,
                'text': chunk
            })
        update_index(vectors, metadata)
        return jsonify({"message": f"PDF processed and indexed successfully. {len(chunks)} chunks created."}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        user_prompt = data.get('prompt', '')
        logger.debug(f"User query: {user_prompt}")

        index = faiss.read_index(INDEX_PATH)
        with open(INDEX_PATH + '.metadata', 'rb') as f:
            metadata = pickle.load(f)

        search_results = search_documents(user_prompt, index, metadata)
        logger.debug(f"Search results: {search_results}")

        if not search_results:
            return jsonify({'generated_text': "No relevant information found in the index."})

        context = ""
        for doc, score in search_results[:3]:  # Use top 3 results
            context += f"Document '{doc['filename']}' (chunk {doc['chunk_id']}, score: {score:.4f}): {doc['text']}\n\n"

        logger.debug(f"Context: {context[:500]}...")

        full_prompt = f"{SYSTEM_MESSAGE}\n\nContext: {context}\n\nUser: {user_prompt}\n\nChatbot:"
        input_ids = tokenizer.encode(full_prompt, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=150, temperature=0.7, top_p=0.9)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        chatbot_response = generated_text.split("Chatbot:")[-1].strip()

        logger.debug(f"Generated response: {chatbot_response[:200]}...")

        return jsonify({'generated_text': chatbot_response, 'context': context})
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/index_info', methods=['GET'])
def index_info():
    try:
        index = faiss.read_index(INDEX_PATH)
        with open(INDEX_PATH + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        return jsonify({
            'total_vectors': index.ntotal,
            'dimension': index.d,
            'total_chunks': len(metadata),
            'sample_chunks': metadata[:5] if metadata else []
        })
    except Exception as e:
        logger.error(f"Error in index_info: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_index_exists()
    processed_files = process_pdf_directory()
    if processed_files == 0:
        logger.warning("No PDF files were processed. The index might be empty.")
    app.run(host='0.0.0.0', port=5001)
