import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def tokenize(text):
    return word_tokenize(remove_punctuation(text.lower()))

# Mecierz term-dokument
def build_term_document_matrix(documents, terms):
    """Tworzy macierz term-dokument."""
    matrix = np.zeros((len(terms), len(documents)))
    for j, doc in enumerate(documents):
        for i, term in enumerate(terms):
            matrix[i, j] = 1 if term in doc else 0
    return matrix

def cosine_similarity(vec1, vec2):
    """Oblicza podobieństwo cosinusowe między dwoma wektorami."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return np.round(dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0, 2)

def lsi_similarity(documents, query, k):
    """
    Realizuje algorytm LSI: redukcję wymiarów i obliczenie podobieństwa
    między dokumentami a zapytaniem.
    """

    tokenized_docs = [tokenize(doc) for doc in documents]
    tokenized_query = tokenize(query)

    # Zbiór unikalnych termów
    all_terms = sorted(set(term for doc in tokenized_docs for term in doc))
    
    term_doc_matrix = build_term_document_matrix(tokenized_docs, all_terms)

    # Dekompozycja SVD
    U, Sigma, VT = np.linalg.svd(term_doc_matrix, full_matrices=False)

    # Redukcja do rzędu k
    Sigma_k = np.diag(Sigma[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    
    # Macierz dokumentów w przestrzeni zredukowanej
    reduced_docs = np.dot(Sigma_k, VT_k)

    # Redukcja zapytania do przestrzeni zredukowanej
    query_vector = np.array([1 if term in tokenized_query else 0 for term in all_terms])
    reduced_query = np.dot(np.linalg.inv(Sigma_k), np.dot(U_k.T, query_vector))
    
    similarities = []
    for i in range(len(documents)):
        similarity = cosine_similarity(reduced_query, reduced_docs[:, i])
        similarities.append(similarity)
    
    return similarities
 
n = int(input())
documents = [input() for i in range(n)]
query = input()
k = int(input())

results = lsi_similarity(documents, query, k)
results = [float(value) for value in results]
print(results)