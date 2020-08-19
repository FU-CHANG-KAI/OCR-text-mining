from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



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
