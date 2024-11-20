import numpy as np
from data.skills import skills_keywords
import spacy
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from models.category_mapper import category_mapping
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from helpers.subhelpers import HumanPrompt
from helpers.subhelpers import OutputClarifier
from helpers.subhelpers import extract_text_from_pdf
from helpers.subhelpers import cleanResume
from helpers.subhelpers import preprocess_text

nlp = spacy.load("en_core_web_sm")
GROQ_API_KEY="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"

clf_path = os.path.join(os.getcwd(), "models", "clf.pkl")
tfidf_path = os.path.join(os.getcwd(), "models", "tfidf.pkl")
clf = pickle.load(open(clf_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))


chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

 

def ResumeParser(pdf_path):
    extText=extract_text_from_pdf(pdf_path)
    cleaned_resume = cleanResume(extText)
    processedText=preprocess_text(extText)
    text = nlp(processedText)
    system = "You are a highly skilled data extractor, capable of identifying and extracting specific details from parsed resumes."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", HumanPrompt(cleaned_resume))])
    chain = prompt | chat
    result = chain.invoke({})
    clarifified_ouput=OutputClarifier(result)
    text2=re.sub(r'\s+', ' ', extText)
    text2 = nlp(text2)
    data = {"name": None, "email": None, "phone": None, "skills": [],"education":[],"expierence":[]}
    skillsSet=set()
    for ent in text.ents:
            if ent.label_ == "PERSON":
                if(len(ent.text.split(" "))>1):            
                    data["name"] = ent.text
                    break
    for ent1 in text2.ents:
            if ent1.label_ == "GPE":
                data["location"] = ent1.text
                
            if(data["name"]==None and len(ent1.text.split(" "))>1):            
                data["name"] = ent1.text
                break
    
                
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', extText)
    if email_match:
        data["email"] = email_match.group(0)

    # Extract Phone Number
    phone_match = re.search(r'(\+?\d{1,2})?\s?(\d{10})', extText)
    if phone_match:
        number = phone_match.group(0).replace("\n", "")
        data["phone"] = number
        
    for skill in processedText.split(" "):
        if skill in skills_keywords:
            skillsSet.add(skill)
    data["skills"] = list(skillsSet)
    data["name"]=clarifified_ouput["name"]        
    data["location"]=clarifified_ouput["location"] 
    data["education"]=clarifified_ouput["education"] 
    data["expierence"]=clarifified_ouput["expierence"] 
    return data


def normalize_distance_to_rank(distance, min_distance, max_distance):
    """
    Normalize the distance to a rank between 1 and 100.
    Smaller distances should give a higher rank (closer to 100).
    """
    # Avoid division by zero and ensure the rank is between 1 and 100
    normalized_rank = 100 - ((distance - min_distance) / (max_distance - min_distance)) * 99
    return max(1, min(100, normalized_rank))

import numpy as np
def classifier(resume):
    cleaned_resume = cleanResume(resume)
    system = "You are a highly skilled data extractor, capable of identifying and extracting specific details from parsed resumes."
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", HumanPrompt(cleaned_resume))])
    chain = prompt | chat
    result = chain.invoke({})
    clarifified_ouput = OutputClarifier(result)

    # Perform classification
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")
    
    # Get the nearest neighbors of the predicted class (for the current input)
    distances, indices = clf.kneighbors(input_features, n_neighbors=5)  # Adjust n_neighbors as needed
    
    # Calculate the approximate centroid of the predicted class
    nearest_neighbor_features = clf._fit_X[indices[0]]  # The training data corresponding to the nearest neighbors
    predicted_centroid = nearest_neighbor_features.mean(axis=0)  # Mean of the nearest neighbors

    # Compute the distance from the input feature to this "centroid"
    distance_to_centroid = np.linalg.norm(input_features.toarray() - predicted_centroid)
    rank = normalize_distance_to_rank(distance_to_centroid, 1, 100)
    

    # Return the classification result along with the distance to centroid
    return {
        "category_name": category_name,
        "name": clarifified_ouput["name"],
        "rank": rank-8
    }

