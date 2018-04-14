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
import pandas as pd


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
    sms = []
    for row in reader:
        if len(row[2]) != 0:
            sms_stemmed.append(row[2])
            sms.append(row[1])
            if row[0] == "spam":
                classification.append(1)
            elif row[0] == "ham":
                classification.append(0)
    sms_stemmed = sms_stemmed[1:]
    sms = sms[1:]
    print(len(sms_stemmed), len(classification))


    # Changing to a bag of words representation
    vectorizer = CountVectorizer()
    x_bow = vectorizer.fit_transform(sms_stemmed)
    print('Shape of x_bow is', x_bow.shape)
    x_bow = x_bow.toarray()

    features = [100,200,300,400,500,1000,1500,2000,2500,3000,3500,4000,4500, 5000, 5500]
    scores = []
    for i in features:
        # Changing to tfidf representation
        from sklearn.feature_extraction.text import TfidfVectorizer
        max_features = i
        tfidf = TfidfVectorizer(max_features=max_features)
        x_tfidf = tfidf.fit_transform(sms_stemmed).toarray()
        classification = np.asarray(classification)

        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.30, random_state=20)

        print('The shape of the training and testing sets are:',X_train.shape, X_test.shape)

        # empty list that will hold cv scores
        cv_acc_scores = []
        cv_prec_scores = []
        # perform 10-fold cross validation
        knn = KNeighborsClassifier(n_neighbors=1)
        acc_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        prec_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='precision')
        scores.append([i, round(acc_scores.mean(), 6), round(prec_scores.mean(), 6)])

    scores_df = pd.DataFrame(scores, columns= ['Max Features', 'Accuracy', 'Precision'])
    scores_df.to_csv('knn max features again_v2.csv')

    # # initiating, fitting and predicting the knn model
    # knn = KNeighborsClassifier(n_neighbors=optimal_k)
    # knn.fit(X_train, y_train)
    # prediction = knn.predict(X_test)
    #
    # print('Accuracy of the model is', accuracy_score(y_test, prediction))
    # print('Precision of the model is', precision_score(y_test, prediction))
    #
    # import winsound
    #
    # frequency = 2500  # Set Frequency To 2500 Hertz
    # duration = 1000  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)