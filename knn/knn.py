from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# def euclidean_dist(a,b):
#     a = np.array(a)
#     b = np.array(b)
#     return np.linalg.norm(a - b)

if __name__ == '__main__':
    #Reading in the file via csv library
    filepath = 'C:\\Users\\Joash\\Desktop\\University Stuff\\4B uni stuff\\SYDE 522\\522 Project\\SMS_spam_or_ham' \
               '\\spam_result'
    csvfile = open(filepath + '.csv', "rt", encoding="utf8")
    reader = csv.reader(csvfile)
    sms_stemmed = []
    classification = []
    for row in reader:
        sms_stemmed.append(row[2])
        if row[0] == "spam":
            classification.append(1)
        elif row[0] == "ham":
            classification.append(0)
    sms_stemmed = sms_stemmed[1:]
    print(len(sms_stemmed), len(classification))

    # Changing to a bag of words representation
    vectorizer = CountVectorizer()
    x_bow = vectorizer.fit_transform(sms_stemmed)
    print('Shape of x_bow is', x_bow.shape)
    x_bow = x_bow.toarray()

    # Changing to tfidf representation
    tfidf = TfidfTransformer()
    x_tfidf = tfidf.fit_transform(x_bow)
    print('Shape of x_tfidf is',x_tfidf.shape)

    # # Euclidean Distance matrix for bow and tdidf
    # e_dist_bow = euclidean_distances(x_bow)
    # e_dist_tdidf = euclidean_distances(x_tfidf)
    # print(e_dist_bow[0,2])
    # print(e_dist_tdidf[0,2])
    #
    # print(round(x_tfidf.shape[0]*.8,0))


    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.33, random_state=42)

    print('The shape of the training and testing sets are:',X_train.shape, X_test.shape)

    # initiating, fitting and predicting the knn model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)

    print('Accuracy of the model is', accuracy_score(y_test, prediction))
    print('Precision of the model is', precision_score(y_test, prediction))

    # creating odd list of K for KNN
    k_list = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]


    # empty list that will hold cv scores
    cv_acc_scores = []
    cv_prec_scores = []

    # perform 10-fold cross validation
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        acc_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        prec_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='precision')
        print(prec_scores)
        cv_acc_scores.append(acc_scores.mean())
        cv_prec_scores.append(prec_scores.mean())

    print(cv_acc_scores)
    print(cv_prec_scores)
    #
    # model accuracy error
    acc_err = [1 - x for x in cv_acc_scores]
    prec_err = [1 - x for x in cv_prec_scores]

    # # determining best k
    # # optimal_k = neighbors[MSE.index(min(MSE))]
    # # print("The optimal number of neighbors is %d" % optimal_k
    #
    # plot misclassification error vs k
    plt.plot(k_list, acc_err)
    plt.plot(k_list, prec_err, color='g')
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()


    # # Manhattan Distance matrix for bow and tdidf
    # m_dist_tdidf = manhattan_distances(x_tfidf)
    # m_dist_bow = manhattan_distances(x_bow)
    # print(m_dist_bow[0,2])
    # print(m_dist_tdidf[0,2])
    #
    # # Gets the feature names in the matrix
    # feature_names = vectorizer.get_feature_names()
    #
    # # Sum up the counts of each vocabulary word from the bag of words representation
    # word_sum = np.sum(x_bow, axis=0)
