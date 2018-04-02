from sklearn.feature_extraction.text import CountVectorizer
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense


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

    from sklearn.feature_extraction.text import TfidfVectorizer
    max_features = 2000
    tfidf = TfidfVectorizer(max_features=max_features)
    x_tfidf = tfidf.fit_transform(sms_stemmed).toarray()
    classification = np.asarray(classification)
    print(type(x_tfidf), x_tfidf.shape, classification.shape)  # 5572 doc, tfidf 100 dimension
    # print(x_tfidf)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.33, random_state=13)

    # build model
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation

    model = Sequential()
    model.add(Dense(32, input_shape=(max_features,)))
    # model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(64))
    # model.add(Dropout(0.2))
    model.add(Activation('relu'))
    # model.add(Dense(32))
    # model.add(Dropout(0.2))
    # model.add(Activation('relu'))
    # model.add(Dense(264))
    # model.add(Dropout(0.2))
    # model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(X_train, y_train, batch_size=32, epochs=30, verbose=1, validation_split=0.2)

    test_pred = model.predict(X_test)
    print(test_pred)
    predicted_classes = np.around(test_pred, decimals=0)
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, predicted_classes)
    print(cm)
    accuracy = (cm[0,0] + cm[1,1])/X_test.shape[0]
    precision = cm[0,0]/(cm[0,0] + cm[1,0])
    print('The accuracy and precision of the model is:', round(accuracy,3), 'and', round(precision,2))

    # print(type(test_pred))
    # predicted_classes = np.around(test_pred, decimals=0)
    # for i in predicted_classes:
    #     print(i)





    # # Changing to a bag of words representation
    # vectorizer = CountVectorizer()
    # x_bow = vectorizer.fit_transform(sms_stemmed)
    # print('Shape of x_bow is', x_bow.shape)
    # x_bow = x_bow.toarray()
    #
    # # Changing to tfidf representation
    # tfidf = TfidfTransformer()
    # x_tfidf = tfidf.fit_transform(x_bow)
    # print('Shape of x_tfidf is',x_tfidf.shape)
    # num_classes = 2
    #
    # # split into train and test
    # X_train, X_test, y_train, y_test = train_test_split(x_tfidf, classification, test_size=0.33, random_state=42)
    #
    # print('The shape of the training and testing sets are:',X_train.shape, X_test.shape)
    # # print(type(y_train))
    # # print(type(X_train))
    # #
    # y_train = np.asarray(y_train)
    # X_train = np.asarray(X_train)
    # X_test = np.asarray(X_test)
    # y_test = np.asarray(y_test)
    # #
    # # print(type(y_train))
    # # print(X_train)
    #
    # # Initializing Neural Network
    # model = Sequential()
    #
    # # Adding the input layer and the first hidden layer
    # # model.add(Dense(activation="relu", input_dim=x_tfidf.shape[1], units=11, kernel_initializer="uniform"))
    # # model.add(Dense(11, activation="relu", input_dim=(5824,1), kernel_initializer="uniform"))
    # model.add(Dense(11, activation='relu', input_dim=(1,5824)))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1)
    #
    # # model.add(Dense(1, activation="relu", input_dim=(1,11), kernel_initializer="uniform"))
    #
    # # model.add(Dense(1, num_classes, activation='softmax'))
    #
    # # Compiling Neural Network
    # # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # # Fitting our model
    # # model_train = model.fit(X_train, y_train, batch_size=5 , epochs=5, verbose=1, validation_data=(X_test, y_test))
    #
    #
    # # # Predicting the Test set results
    # # y_pred = classifier.predict(X_test)
    # # y_pred = (y_pred > 0.5)
    # #
    # # # Creating the Confusion Matrix
    # # from sklearn.metrics import confusion_matrix
    # #
    # # cm = confusion_matrix(y_test, y_pred)