import numpy as np 
from numpy.linalg import norm
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def eucledian_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = np.array(A), np.array(B)
    sum = np.sum(np.square(A-B))
    return np.sqrt(sum)

def cosine_similarity(A, B):
    A, B = np.array(A), np.array(B)
    cosine = np.dot(A, B) / (norm(A)* norm(B))
    return cosine

def hamming_distance(s1, s2):
    s1, s2 = np.array(s1), np.array(s2)
    dist = 0
    if len(s1) != len(s2):
        return -1       
    for i, j in zip(s1,s2):
        if i != j:
            dist += 1 
    return dist

def manhattan_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A, B = np.array(A), np.array(B)
    sum = np.sum(np.absolute(A-B))
    return sum


def chebyshev_distance(A: np.ndarray, B: np.ndarray) -> np.array:
    A, B = np.array(A), np.array(B)
    sum = np.max(np.absolute(A-B))
    return sum


def minkowski_distance(A: np.ndarray, B: np.ndarray, h: int) -> np.ndarray:
    A, B = np.array(A), np.array(B)
    sum = np.sum(np.power(A-B,h))
    return np.power(sum, 1/h) 
    

def jaccard_similarity(s1, s2):
    # Tokenizing sentences, i. e., 
    # splitting the sentences into 
    # words
    s1_list = word_tokenize(s1)
    s2_list = word_tokenize(s2)
    # Getting the English stopword collection
    sw = stopwords.words('english') 
    # Creating word sets corresponding to each sentence
    S1_set = {word for word in s1_list if not word in sw}
    S2_set = {word for word in s2_list if not word in sw}

    print(f'Word set Sentence 1 = {S1_set}') 
    print(f'Word set Sentence 2 = {S2_set}')

    I= set(S1_set).intersection(set(S2_set))
    U= set(S1_set).union(set(S2_set))
    print(f'Intersection = {I}') 
    print(f'Union = {U}')
    #intersection over union
    IoU = len(I)/len(U)
    return IoU