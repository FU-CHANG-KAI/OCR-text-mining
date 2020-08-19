from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from doc2vec import doc2vec, doc2vec_to_vectors


def kmeans(featureVectors, model):  
    sum_of_distance = []  

    for i in range(1,24):
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(featureVectors)
        sum_of_distance.append(kmeans.inertia_)

    fig = plt.figure()
    plt.plot(range(1, 24), sum_of_distance)
    plt.scatter(range(1, 24), sum_of_distance, color='g',marker='x')
    plt.grid(True)
    plt.xlabel('K')
    plt.ylabel('Sum of square distance to centroids')
    plt.title('Elbow curve for {} vectors'.format(model))
    plt.savefig('kmeans+{}'.format(model))

if __name__ == "__main__":
    doc = pickle_file('./tfn/output/books.pkl')
    #Dataset("doc2vec")  
    model = doc2vec(doc, 100, 300, 0.025)
    pickle_file.save(model, './tfn/output/model-100-300-0.025')
    d2v_vectors = doc2vec_to_vectors(model)
    