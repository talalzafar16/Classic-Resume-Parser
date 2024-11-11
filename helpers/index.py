from pdfminer.high_level import extract_text
from docx import Document
from data.skills import skills_keywords
import spacy
import re
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    text = extract_text(pdf_path)
    return text
    
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
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
