def plagiarism_check(text1, text2):
    # Get the ratio of similarity
    similarity_ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return similarity_ratio

text1 = """This is a sample text for plagiarism detection."""
text2 = """This is a sample text for checking plagiarism."""

similarity = plagiarism_check(text1, text2)
print(f"Similarity Ratio: {similarity:.2f}")

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

text1 = """This is a sample text for plagiarism detection."""
text2 = """This is a sample text for checking plagiarism."""

similarity = cosine_plagiarism_check(text1, text2)
print(f"Cosine Similarity: {similarity:.2f}")

if similarity > 0.7:  # threshold for plagiarism
    print("Plagiarism detected!")
else:
    print("No plagiarism detected.")


import spacy

nlp = spacy.load('en_core_web_md')  # Load a medium-sized English model

def spacy_plagiarism_check(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

text1 = """This is a sample text for plagiarism detection."""
text2 = """This is a sample text for checking plagiarism."""

similarity = spacy_plagiarism_check(text1, text2)
print(f"Semantic Similarity: {similarity:.2f}")

if similarity > 0.7:  # threshold for plagiarism
    print("Plagiarism detected!")
else:
    print("No plagiarism detected.")
