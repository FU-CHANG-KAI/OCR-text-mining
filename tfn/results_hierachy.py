import pickle_file
from models import hierarchy

from feature_extraction import doc2vec
from feature_extraction import tf_idf


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS


books = pickle_file.load('books.pkl')
doc = books['contents'][1:].tolist() 

# Use doec2vec to perform feature extracture
feature_vectors = pickle_file.load('doec2vec.pkl')


# Use tf-idf to extract the features and calculate the dissimilarity bt cosine simularity
vectorizer, corpus_matrix, feature_names = tf_idf.get_tfidf_model(doc)
distVectors = 1-cosine_similarity(corpus_matrix)

mds = MDS(n_components=2000, random_state=1,dissimilarity="precomputed", metric=True)
bookFeatures = mds.fit_transform(distVectors)


# output using feature extractor: doc2vec
hierarchy.ward_cluster(6, feature_vectors, 'doc2vec' )

# output using feature extractor: tf-idf
hierarchy.ward_cluster(6, bookFeatures, 'tf-idf')





