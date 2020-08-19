import pickle_file
import list_to_csv 
from models import kmeans
from feature_extraction import doc2vec
from feature_extraction import tf_idf

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS


books = pickle_file.load('books.pkl')
doc = books['contents'][1:].tolist()

# Use doec2vec to perform feature extracture
model = doc2vec.doc2vec(doc, 300, 100, 0.025)
feature_vectors = doc2vec.doc2vec_to_vectors(model)
#print('Convert doc2vec vector to a csv file')
#list_to_csv.d2vm_vec(feature_vectors)

pickle_file.save(feature_vectors, 'doc2vec_stemm_300.pkl')

# Perform KMeans with doc2vec features
model = 'doc2vec'
kmeans.kmeans(feature_vectors, model)
# doc2vec already generate the similarity vector, no need to call cosine similarity


# Use tf=idf to perform feature extraction
vectorizer, corpus_matrix, feature_names = tf_idf.get_tfidf_model(doc)
distVectors = 1-cosine_similarity(corpus_matrix)

mds = MDS(n_components=2, random_state=1,dissimilarity="precomputed", metric=True)
bookFeatures = mds.fit_transform(distVectors)

model = 'tf-idf'
kmeans.kmeans(distVectors, model)