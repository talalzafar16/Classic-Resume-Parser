from flask import Flask
import os
from helpers.index import ResumeParser

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Its Test Api!"

@app.route('/extract-text',methods=["GET"])
def Parse():
    pdf_path1 = os.path.join(os.getcwd(), "data", "resumes", "resume2.pdf")
    pdf_path2 = os.path.join(os.getcwd(), "data", "resumes", "Anas Khan CV (1).pdf")
    pdf_path3 = os.path.join(os.getcwd(), "data", "resumes", "resume1.pdf")
    a=ResumeParser(pdf_path1)
    b=ResumeParser(pdf_path2)
    c=ResumeParser(pdf_path3)
    return {"A":a,"B":b,"C":c}



if __name__ == "__main__":
    app.run(debug=True)