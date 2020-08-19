import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def ward_cluster(n, bookFeatures, model):
    #parameter setting
    #Agglomerative Clustering method
    model = AgglomerativeClustering(n_clusters = n, linkage = 'ward')


    model.fit(bookFeatures)
    labels = model.fit_predict(bookFeatures)


    #results visualization
    plt.figure()
    plt.scatter(bookFeatures[:,0], bookFeatures[:,1], c = labels)
    plt.axis('equal')
    plt.title('Hierarchical Clustering(Ward) for {} features'.format(model))
    #plt.show()
    plt.savefig('Ward clustering _{}'.format(model))
    plt.clf()
    

    # Performs hierarchical/agglomerative clustering on X by using "Ward's method"
    
    linkage_matrix = linkage(bookFeatures, 'ward')
    figure = plt.figure(figsize=(7.5, 5))
    # Plots the dendrogram
    dendrogram(linkage_matrix, labels = labels)
    plt.title('Hierarchical Clustering Dendrogram (Ward) for {} features'.format(model) )
    plt.xlabel('cluster index')
    plt.ylabel('distance')
    plt.tight_layout()
    #plt.show()
    plt.savefig('Hierarchical Clustering Dendrogram (Ward)')
    plt.clf()