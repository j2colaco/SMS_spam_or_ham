from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
from sklearn.metrics import confusion_matrix
import winsound
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import time

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

    print(len(sms_stemmed), len(classification))
    # sms_pd = pd.DataFrame(sms_stemmed)
    # sms_pd.to_csv('check.csv')
    random_state = 2
    pre_score = []
    acc_score = []
    scores = []
    predicted_classes = []
    test_pred = []

    X_tr, X_te, y_tr, y_te = train_test_split(sms_stemmed, classification, test_size=0.30, random_state=random_state)

    max_features = 1500
    tfidf = TfidfVectorizer(max_features=max_features)
    x_tfidf = tfidf.fit_transform(sms_stemmed).toarray()
    classification = np.asarray(classification)
    print(type(x_tfidf), x_tfidf.shape, classification.shape)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.30, random_state=random_state)
    print(len(X_test), len(y_test))


    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    count = 1

    for train, validate in kfold.split(X_train, y_train):
        print('This is', count, 'fold!')
        model = Sequential()
        model.add(Dense(60, input_shape=(max_features,)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(5))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        model.fit(X_train[train], y_train[train], batch_size=64, epochs=10, verbose=1)

        test_pred = model.predict(X_train[validate])
        # print(test_pred)
        predicted_classes = np.around(test_pred, decimals=0)
        # Creating the Confusion Matrix
        cm = confusion_matrix(y_train[validate], predicted_classes)
        print(cm)
        accuracy = (cm[0, 0] + cm[1, 1]) / X_train[validate].shape[0]
        precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        print('The accuracy and precision of the model is:', round(accuracy, 3), 'and', round(precision, 3))
        acc_score.append(accuracy)
        pre_score.append(precision)
        count += 1

    print('Model accuracy is', round(sum(acc_score) / len(acc_score),3))
    print('Model precision is', round(sum(pre_score) / len(pre_score),3))
    t1 = time.time()
    test_pred = model.predict(X_test)
    predicted_classes = np.around(test_pred, decimals=0)
    t2 = time.time()
    print(t2 - t1)
    cm = confusion_matrix(y_test, predicted_classes)
    print(cm)
    # accuracy = (cm[0, 0] + cm[1, 1]) / X_test.shape[0]
    accuracy = accuracy_score(y_test, predicted_classes)
    # precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    precision = precision_score(y_test, predicted_classes)
    print('Test accuracy is', accuracy)
    print('Test precision is', precision)

    print(len(X_test), len(test_pred))
    test_output = []
    for i in range(0, len(X_test), 1):
        test_output.append([X_te[i], y_test[i], predicted_classes[i][0]])

    test_output_df = pd.DataFrame(test_output, columns=['sms_stemmed', 'Actual Classification', 'Predicted Classification'])
    test_output_df.to_csv('output.csv', index=False)

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 500  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)