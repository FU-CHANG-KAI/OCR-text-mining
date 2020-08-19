from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess import Dataset
import pickle_file

def doc2vec(doc, vec_size, max_epoch=100, alpha=0.025):
    ## prepare doc2vec input - list of taggedDocument
    tagged_doc = []
    for index, text in enumerate(doc):
        tagged_doc.append(TaggedDocument(words=text, tags=[str(index)]))
    # the document has been tokenized, no need to preprocess again
    # setup configurations
    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    model.build_vocab(tagged_doc)

    # Train the model
    print("Training doc2vec model..")
    for epoch in range(max_epoch):
        #print('iteration {0}'.format(epoch))
        model.train(tagged_doc, total_examples=model.corpus_count, epochs=model.iter )
        # Change learning rate for next epoch (start with large num to speed up 
        # at first and then decrease to fine grain learning)

        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save("d2v.model")
    print("Done training..")
    return model

def doc2vec_to_vectors(model): 
    # Extract vectors from doc2vec model
    feature_vectors = []
    print(len(model.docvecs))
    for i in range(0,len(model.docvecs)) :
        feature_vectors.append(model.docvecs[i])
    
    return feature_vectors    




        

