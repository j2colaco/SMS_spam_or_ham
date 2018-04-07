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
    from sklearn.feature_extraction.text import TfidfVectorizer
    max_features = 350
    tfidf = TfidfVectorizer(max_features=max_features)
    x_tfidf = tfidf.fit_transform(sms_stemmed).toarray()
    classification = np.asarray(classification)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.30, random_state=42)

    print('The shape of the training and testing sets are:',X_train.shape, X_test.shape)

    # creating odd list of K for KNN
    # k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    k_list = [1]

    # empty list that will hold cv scores
    cv_acc_scores = []
    cv_prec_scores = []
    count = 0
    # perform 10-fold cross validation
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k)
        acc_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        prec_scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='precision')
        cv_acc_scores.append(round(acc_scores.mean(), 3))
        cv_prec_scores.append(round(prec_scores.mean(), 3))
        print("Testing out different k's", count)
        count += 1

    print(cv_acc_scores)
    print(cv_prec_scores)

    # model accuracy error
    acc_err = [1 - x for x in cv_acc_scores]
    prec_err = [1 - x for x in cv_prec_scores]

    # # plot misclassification error vs k
    # plt.plot(k_list, acc_err, color='b', label='Accuracy Error')
    # plt.plot(k_list, prec_err, color='g', label='Precision Error')
    # plt.xlabel('Nearest Neighbors K')
    # plt.ylabel('Model Error')
    # plt.legend()
    # plt.title('K vs Model Error (Raw Input)')
    # plt.show()

    # Optimal K
    min_value = min(acc_err)
    # print(min_value)
    min_index = acc_err.index(min_value)
    # print(min_index)
    optimal_k = k_list[min_index]
    print('K value with the lowest accuracy error is', optimal_k)

    # initiating, fitting and predicting the knn model
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)

    print('Accuracy of the model is', accuracy_score(y_test, prediction))
    print('Precision of the model is', precision_score(y_test, prediction))

    import winsound

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)