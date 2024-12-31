from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Flask App Initialization
app = Flask(__name__)
# API Key Configuration
API_KEY = "AIzaSyBrvAp2LifjXfRax0y-MOIO-_Iboscoo0w"  # Replace with your valid API key
genai.configure(api_key=API_KEY)

# Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the dataset
DATASET_PATH = "data/response.csv"  # Replace with your dataset file path
data = pd.read_csv(DATASET_PATH, encoding="utf-8")

# Load SentenceTransformer model for semantic search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for the `query` column
print("Generating embeddings for dataset...")
data["embedding"] = data["query"].apply(lambda x: embedding_model.encode(str(x), convert_to_tensor=False))

# Convert embeddings into a numpy array
embedding_matrix = np.vstack(data["embedding"].to_numpy())

# Create FAISS index
embedding_dim = embedding_matrix.shape[1]  # Dimension of embeddings
faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (euclidean) for similarity search
faiss_index.add(embedding_matrix)  # Add dataset embeddings to the FAISS index

print(f"FAISS index created with {faiss_index.ntotal} entries.")

def clean_response(response_text):
    """
    Cleans and formats the chatbot's output for better readability.
    - Removes unnecessary asterisks.
    - Adds proper line breaks and spacing.
    - Formats lists and sections for clarity.
    """
    response_text = response_text.replace("*", "").replace(". ", ".\n\n")
    response_text = response_text.replace("Suggestions:", "**Suggestions:**\n\n-")
    return "\n\n".join([line.strip() for line in response_text.split("\n\n") if line.strip()])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get data from the AJAX request
        symptoms = request.json.get("symptoms", "")
        duration = request.json.get("duration", "")
        age = request.json.get("age", "")
        sex = request.json.get("sex", "")

        # User's input query
        user_query = f"Symptoms: {symptoms}, Duration: {duration}, Age: {age}, Sex: {sex}"

        # Define keywords related to mental health
        keywords = [
            "therapy", "therapist", "mental health", "depression", "anxiety", 
            "stress", "emotional", "well-being", "psychological", "condition", 
            "sad", "sadness", "feel", "feeling", "mood", "overwhelmed", "help"
        ]

        # Check if any keyword is in the symptoms or other fields
        if not any(keyword in symptoms.lower() or keyword in user_query.lower() for keyword in keywords):
            return jsonify({"response": "As a therapist chatbot, I am here to assist you. Please feel free to share any queries related to mental health, emotional well-being, or psychological concerns."})

        # Compute the embedding for the user's query
        user_embedding = embedding_model.encode(user_query, convert_to_tensor=False).reshape(1, -1)

        # Search the FAISS index for the most similar query
        k = 1  # Number of top results to retrieve
        distances, indices = faiss_index.search(user_embedding, k)

        # Retrieve the most similar result
        best_match_idx = indices[0][0]
        best_similarity = distances[0][0]

        if best_similarity < 0.5:  # Set a similarity threshold
            # Fallback to Gemini for a generated response
            prompt = (
                f"You are a medical chatbot acting as a therapist. A user has provided symptoms, duration, age, and gender. "
                f"Suggest possible lifestyle advice or actions without giving a formal diagnosis or using medical terms. "
                f"User 's Input: {user_query}"
            )
            response = model.generate_content(prompt)
            final_response = clean_response(response.text)
        else:
            # Retrieve response from the dataset
            response_text = data.iloc[best_match_idx]["Empathetic Response"]

            # Use Gemini to refine the response
            prompt = ("You are a medical chatbot acting as a therapist. Refine the following response to make it more empathetic, "
            f"professional, and user-friendly. Avoid medical diagnoses or technical terms: {response_text}"
            )
            response = model.generate_content(prompt)
            final_response = clean_response(response.text)

        return jsonify({"response": final_response})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)