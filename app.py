import nltk
nltk.download('stopwords')
import spacy
nlp = spacy.load("en_core_web_sm")
from flask import Flask, request, jsonify
import os
from helpers.index import ResumeParser
import pickle
from models.category_mapper import category_mapping
# from pyresparser import ResumeParser

# Load pre-trained models
clf_path = os.path.join(os.getcwd(), "models", "clf.pkl")
tfidf_path = os.path.join(os.getcwd(), "models", "tfidf.pkl")
clf = pickle.load(open(clf_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, it's Test API!"

@app.route('/get-parsed-data', methods=["POST"])
def parse():
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
    
    uploaded_files = request.files.getlist('files')  
    resumes_data = []  
    
    for uploaded_file in uploaded_files:
        if uploaded_file.filename == '':
            continue
        # Save file temporarily for processing
        temp_path = os.path.join(os.getcwd(), "temp", uploaded_file.filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        uploaded_file.save(temp_path)
        
        # Parse the resume
        try:
            parsed_data = ResumeParser(temp_path)
            resumes_data.append({
                "filename": uploaded_file.filename,
                "name": parsed_data.get('name', []),
                "email": parsed_data.get('email', []),
                "phone": parsed_data.get('phone', []),
                "location": parsed_data.get('location', []),
                "skills": parsed_data.get('skills', [])
            })
        except Exception as e:
            resumes_data.append({
                "filename": uploaded_file.filename,
                "error": str(e)
            })
        finally:
            # Clean up the temporary file
            os.remove(temp_path)

    return jsonify({"resumes": resumes_data})

if __name__ == "__main__":
    app.run(debug=True)
