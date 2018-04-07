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
    # sms_pd = pd.DataFrame(sms_stemmed)
    # sms_pd.to_csv('check.csv')


    features = [100,200,300,400,500,600,700,800,900,1000, 1500, 2000, 2500, 3000,3500,4000,4500,5000]
    features = [1000]
    # features = [2500, 3000,3500,4000,4500,5000]
    pre_score = []
    acc_score = []
    scores = []

    for i in features:
        print('On', i, 'MAX FEATURES!')
        from sklearn.feature_extraction.text import TfidfVectorizer
        max_features = i
        tfidf = TfidfVectorizer(max_features=max_features)
        x_tfidf = tfidf.fit_transform(sms_stemmed).toarray()
        classification = np.asarray(classification)
        print(type(x_tfidf), x_tfidf.shape, classification.shape)

        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.30, random_state=13)

        kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        count = 1
        for train, validate in kfold.split(X_train, y_train):
            print('This is', count, 'fold!')
            model = Sequential()
            model.add(Dense(32, input_shape=(max_features,)))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(Dense(64))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.summary()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
            model.fit(X_train[train], y_train[train], batch_size=64, epochs=10, verbose=1)

            test_pred = model.predict(X_train[validate])
            # print(test_pred)
            predicted_classes = np.around(test_pred, decimals=0)
            # Creating the Confusion Matrix
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_train[validate], predicted_classes)
            print(cm)
            accuracy = (cm[0, 0] + cm[1, 1]) / X_train[validate].shape[0]
            precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            print('The accuracy and precision of the model is:', round(accuracy, 3), 'and', round(precision, 3))

            acc_score.append(accuracy)
            pre_score.append(precision)
            count += 1
        scores.append([i, (sum(acc_score)/len(acc_score)), (sum(pre_score)/len(pre_score))])
    print(scores)
    scores_pd = pd.DataFrame(scores, columns=['Max Features', 'Test Accuracy', 'Test Precision'])
    scores_pd.to_csv('Pick Max Features NN_v2.csv', index=False)
    print('Added to file')

    import winsound

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)