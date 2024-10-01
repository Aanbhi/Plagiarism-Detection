# Plagiarism Detection

import difflib

def plagiarism_check(text1, text2):
    similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity_ratio

# Define the texts
text1 = "Natural language processing is an exciting area of AI."
text2 = "The study of artificial intelligence includes natural language processing."

# Call the plagiarism_check function
similarity = plagiarism_check(text1, text2)

# Print the result
print("Similarity:", similarity)

if similarity > 0.7:  # threshold for plagiarism
    print("Plagiarism detected!")
else:
    print("No plagiarism detected.")
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_plagiarism_check(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])
    return cosine_sim[0][0]



import spacy

# Load the SpaCy model
try:
    nlp = spacy.load('en_core_web_md')
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()  # Exit if the model fails to load

# Define the text
text1 = "Natural language processing is an exciting area of AI."

# Process the text
doc1 = nlp(text1)

# Example usage: print tokens
for token in doc1:
    print(token.text, token.pos_, token.dep_)

# Content Relevance and Quality Analysis

# 1. Keyword Extraction

from rake_nltk import Rake
import nltk

# Download required NLTK data
nltk.download('stopwords') 
def extract_keywords(text):
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    r = Rake()
    try:
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
text = "Natural language processing is a fascinating field of study."
keywords = extract_keywords(text)
print("Keywords:", keywords) 

# 2. Sentiment Analysis

from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment

text = "I love programming in Python!"
sentiment = analyze_sentiment(text)
print("Sentiment:", sentiment)

# 3. Readability Scoring

import textstat

def readability_analysis(text):
    readability_score = textstat.flesch_kincaid_grade(text)
    return readability_score

text = "This is an example sentence to analyze readability."
score = readability_analysis(text)
print("Readability Score (Flesch-Kincaid):", score)

# 4. Semantic Similarity

import spacy

# Load the SpaCy model and handle potential errors
try:
    nlp = spacy.load('en_core_web_md')
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()  # Exit the script if the model fails to load

def semantic_similarity(text1, text2):
    # Process both texts
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# Define the texts
text1 = "Natural language processing is an exciting area of AI."
text2 = "The study of artificial intelligence includes natural language processing."

# Calculate semantic similarity
try:
    similarity = semantic_similarity(text1, text2)
    print("Semantic Similarity:", similarity)
except Exception as e:
    print(f"An error occurred while calculating similarity: {e}")
