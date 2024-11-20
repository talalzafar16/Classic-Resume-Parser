import json
import re
from pdfminer.high_level import extract_text

def HumanPrompt(cleaned_resume):
    human = f"""
    Analyze the following parsed CV data and extract the following information:
    1. Name of the person.
    2. Location (city, state, or country).
    3. Educational qualifications (degree, institution, and year of graduation).
    4. Job experience (Job Title, Company, Duration, Key Responsibilities).
    the keys of json should be name, location, education and job
    The extracted details should be in a structured JSON format.
    Only return the JSON object, no text. The output should start with a curly bracket and also end with it, no text.

    Parsed CV Data:
    {cleaned_resume}
    """
    return human

def OutputClarifier(result):
    raw_response = result.content
    if "```" in raw_response:
        content = raw_response.split("```")[1]
    else:
        content = raw_response
    try:
        parsed_content = json.loads(content)
        name = parsed_content.get("name", "Name not found")
        location = parsed_content.get("location", "Location not found")
        education = parsed_content.get("education", "Education not found")
        expierence = parsed_content.get("job", "job not found")
        return {"name":name,"location":location,"education":education,"expierence":expierence}

    except json.JSONDecodeError:
        print("Error: The content is not a valid JSON string.")
        return "Error in extracting details"


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
