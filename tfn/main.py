#import scrapt
import pandas
import numpy
import pickle_file
from sklearn.metrics.pairwise import cosine_similarity
import Dataset    
from feature_extraction.doc2vec import doc2vec
from feature_extraction.doc2vec import doc2vec_to_vectors
from feature_extraction.tf_idf import get_tfidf_model


ds = Dataset()
doc_lemma = ds.X[1:].tolist() # Skip the first row as it is empty due to the walk directory process 
pickle_file.save(doc_lemma, 'doc_lemmatization.pkl')  

doc_lemma = pickle_file.load('doc2vec_lemma_300.pkl')
# Feature extraction by tf_idf and processed by cosine_similarity to find the document similsrity
vec, matrix, feat_names = get_tfidf_model(doc_lemma)
tf_vec_lemma = 1-cosine_similarity(matrix)

# Train the doc2vec model and implement feature extraction process
#  
d2c_300 = doc2vec(doc_lemma, 300, 100, 0.25)