# OCR-data-mining
Optical character recognition (OCR) is a widely used technique to store the old books digitally. However, the recognition error is generated very often in the process of OCR since some of the books are several century ago....
In addition, to categorize and store these old books in a proper way will cost a big human force. 
Thus, natural language processing and data mining techniques become critical to deal with this.
In this repositary, Doc2Vec and TF-IDF are utilized to generate pharagraph vectors. 

TF-IDF has been largely used to process latent semantic analysis with cosine similarity. 
Doc2Vec inherits the two Neural network functions in Word2Vec but and a paragraph ID, which can transform the paragraph vectors to a more complex space i.e. 100 - 400 dimensions for one paragraph. It can also take the order of words in considerration, which is an advantage comparing to TF-IDF. 

<div align=center><img src="image/OCR-flow%20chart.png" width="700"></div>

_Figure.1 FLow chart for OCR Data Mining task_

## Installation
```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
```

## Observation and Conclusion
1. Trade-off of dimension reduction tools (SVD, tSNE, PCA) on TF-IDF paragraph vectors:  MDS might be the best since TF-IDF + cosine similarity require a non-Euclidean dimension reduction as the result of cosine_similarity does not includes the magnitude of textual vectors. Below chart shows SVD is generally better than tSNE according to the 2D visualization.
<div align=center><img src="image/SVD%20mapping_TF-IDF%20vectors.png" width="500"></div>

<div align=center>######Figure.2 SVD mapping TF-IDF vectors</div>

<div align=center><img src="image/tSNE%20mapping_TF-IDF%20vectors.png" width="500"></div>

_Figure.3 tSNE mapping_TF-IDF vectors_

2. Monitor a potentially better cluster number by Elbow curve and Silhouette score

<div align=center><img src="image/kmeans%20clustering%20of%20tf_idf%2BMDS.png" width="500"></div>
_Figure.4 Elbow curve for TF-IDF + MDS _

<div align=center><img src="image/Silhouette%20Score_TF-IDF_MDS.png" width="500"></div>
_Figure.5 Silhouette Score_TF-IDF_MDS _

## Source of data
The data is provided by Msc Data Science at the Uiversity of Southampton. 
I am not permitted to upload the original dataset here.
The dataset is 24 OCR books which contains 300 - 1500 pages. 
All of them were written in English.

## Contributing 
Fell free to contact me at fuchangkai153@gmail.com if you have any query.

我們從這一行開始。

兩個換行符號分開了這行和前面那一行，所以這會變成 **分開的段落** 。

這行也是個分開的段落，不過……
只有一個換行符號分開這行，所以這是 *同段落* 中的分開兩行。
