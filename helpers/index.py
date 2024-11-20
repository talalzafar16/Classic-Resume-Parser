from pdfminer.high_level import extract_text
from data.skills import skills_keywords
import spacy
import re
import pickle
from nltk.corpus import stopwords
from models.category_mapper import category_mapping
from nltk.tokenize import word_tokenize
import nltk
import os
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

clf_path = os.path.join(os.getcwd(), "models", "clf.pkl")
tfidf_path = os.path.join(os.getcwd(), "models", "tfidf.pkl")
clf = pickle.load(open(clf_path, 'rb'))
tfidf = pickle.load(open(tfidf_path, 'rb'))




def remove_stopwords(text, language='english'):
    """
    Remove stopwords from a given text.

    Parameters:
        text (str): The input text from which to remove stopwords.
        language (str): The language of the stopwords. Default is 'english'.

    Returns:
        str: Text without stopwords.
    """
    stop_words = set(stopwords.words(language))
    words = word_tokenize(text)
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
            print(ent.label_,ent.text)
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


def classifier(resume):
    cleaned_resume = cleanResume(resume)
    input_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category_name = category_mapping.get(prediction_id, "Unknown")
    return category_name