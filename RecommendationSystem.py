from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    
    #create a List to store tokens
    arrayList = []

    #Iterate through the list with 'genres' and append it to list
    for mov in movies['genres']:
        #find all values using tokenize_string function and append the return value to the list.
        arrayList.append(tokenize_string(mov))
    #Once the list is ready add new column in movies as tokens.
    movies['tokens'] = arrayList

    #return movies with 'tokens' column
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
     
    #dict to store features
    frequencyFeatures = defaultdict(lambda: 0)
    #dict to store list of features
    listOfFeatures = list(defaultdict(lambda: 0))

    for i in movies['tokens']:
        elements = set(i)
        #Number of unique documents containing each feature
        for e in elements:
            frequencyFeatures[e] += 1
        ftdict = defaultdict(lambda: 0)
        #frequency of each doc
        for j in i:      
            ftdict[j] += 1
        #add values to list
        listOfFeatures.append(ftdict)

    #sort vocab
    features = sorted(frequencyFeatures)
    vocab = dict((l, features.index(l)) for l in features)
    
    #dict to save feature matrix
    finalFeatureMatrix = []

    #number of documents
    N = movies.shape[0]
    
    #cal the value of tfidf
    for i in range(N):
        documentFeature = listOfFeatures[i]
        maxiumFrequency = documentFeature[max(documentFeature, key = documentFeature.get)]
        row = []
        col = []
        #to store tfidf values 
        valueoftfidf = []
        for k in documentFeature:
            if k  in vocab:
                row.append(0)
                col.append(vocab[k])
                #term2 = maxiumFrequency * np.log(N / frequencyFeatures[k])
                tfidf = documentFeature[k] / maxiumFrequency * np.log(N / frequencyFeatures[k])
                #add the calculated value tolist
                valueoftfidf.append(tfidf)
        #create a matrix
        csrMatrix = csr_matrix((valueoftfidf, (row, col)), shape = (1, len(vocab)))
        #once matrix is created append it to finalFeatureMatrix
        finalFeatureMatrix.append(csrMatrix)

    #add new column
    movies['features'] = pd.Series(finalFeatureMatrix, movies.index)

    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
      
    #calculate dot(a,b)
    term1 = np.dot(a, b.T).toarray()[0][0]
    #calculate Euclidean norm of vector a.
    term2 = np.linalg.norm(a.toarray())
    #calculate Euclidean norm of vector b.
    term3 = np.linalg.norm(b.toarray())
    
    
    #Calculate cousine using dot(a, b) / ||a|| * ||b||
    cousineSim = term1 / (term2 * term3)
    
    #return Calculated cousineSim
    return cousineSim


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
     
    #dict to store predicted ratings
    predictedRating = []
    #Calculate predicted ratings for each element of ratings_test
    for i, row in ratings_test.iterrows():
        feat = movies.loc[movies['movieId'] == row['movieId']].squeeze()['features']
        movie_train = ratings_train.loc[ratings_train['userId'] == row['userId']]
        #dict to store cosine list
        cosineList = []
        #variable to store cosine sum
        cosineSum = 0
        for j, row1 in movie_train.iterrows():
            feat1 = movies.loc[movies['movieId'] == row1['movieId']].squeeze()['features']
            #call cousine_sim() to calculate cousine sim value
            cosineSim = cosine_sim(feat, feat1)
            if cosineSim > 0:
                cosineList.append(cosineSim * row1['rating']);
                cosineSum += cosineSim
        if cosineSum > 0:
            predictedRating.append(sum(cosineList) / cosineSum)
        else:
            predictedRating.append(movie_train['rating'].mean())
    #Convert predictedrating to an array
    finalResult = np.array(predictedRating)
    
    #return calculated array
    return finalResult


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()

