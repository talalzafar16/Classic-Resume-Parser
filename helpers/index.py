from pdfminer.high_level import extract_text
from data.skills import skills_keywords
import spacy
import re
import pickle
from models.category_mapper import category_mapping
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
nlp = spacy.load("en_core_web_sm")
GROQ_API_KEY="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"

clf_path = os.path.join(os.getcwd(), "models", "clf.pkl")
tfidf_path = os.path.join(os.getcwd(), "models", "tfidf.pkl")
clf = pickle.load(open(clf_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))


chat = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

def remove_stopwords(text):
    """
    Remove stopwords from a given text.
    Parameters:
        text (str): The input text from which to remove stopwords.
        language (str): The language of the stopwords. Default is 'english'.
    Returns:
        filtered_text (str): Text without stopwords.
    """
    
    stop_words = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", 
    "once", "here", "there", "when", "where", "why", "how", "all", "any", 
    "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
    "t", "can", "will", "just", "don", "should", "now"
    ])
    words = text.split()  # Simple tokenization by spaces
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

def cleanResume(txt):
    """
    Clean the text in the resume i.e. remove unwanted chars in the text. For e.g. 
    1 URLs,
    2 Hashtags,
    3 Mentions,
    4 Special Chars,
    5 Punctuations
    Parameters:
        resume_text (str): The input resume text to be cleaned.
    Returns:
        clean_text (str): Clean Resume.
    """
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    cleanText  = remove_stopwords(cleanText)
    return cleanText

def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text
 

def preprocess_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    # text = text.replace("  ", " ")
    # text=re.sub(r'\s+', ' ', text)
    return text

def ResumeParser(pdf_path):
    extText=extract_text_from_pdf(pdf_path)
    processedText=preprocess_text(extText)
    text = nlp(processedText)
    text2=re.sub(r'\s+', ' ', extText)
    text2 = nlp(text2)
    data = {"name": None, "email": None, "phone": None, "skills": []}
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

        
    return data


import json

def classifier(resume):
    cleaned_resume = cleanResume(resume)

    system = "You are a highly skilled data extractor, capable of identifying and extracting specific details from parsed resumes."
    human = f"""
    Analyze the following parsed CV data and extract the following information:
    1. Name of the person.
    2. Location (city, state, or country).
    3. Educational qualifications (degree, institution, and year of graduation).
    4. Experience (Job Experience)

    The extracted details should be in a structured JSON format.
    Only return the JSON object, no text. The output should start with a curly bracket and also end with it, no text.

    Parsed CV Data:
    {cleaned_resume}
    """

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    result = chain.invoke({})

    raw_response = result.content
    if "```" in raw_response:
        content = raw_response.split("```")[1]
    else:
        content = raw_response

    try:
        parsed_content = json.loads(content)

        # Access the desired fields (name, location, education)
        name = parsed_content.get("name", "Name not found")
        location = parsed_content.get("location", "Location not found")
        education = parsed_content.get("education", "Education not found")

        # Print the extracted fields for verification
        print(f"Name: {name}")
        print(f"Location: {location}")
        print(f"Education: {education}")

    except json.JSONDecodeError:
        print("Error: The content is not a valid JSON string.")
        return "Error in extracting details"

    # Perform classification
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")

    return {"category_name":category_name,"name":name}
