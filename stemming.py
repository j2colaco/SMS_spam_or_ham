import pandas as pd
import csv
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# # Stems words to their root words and removes all characters that are not alpha-numeric
# def stem_str(str):
#     ret_str = ""
#     for w in word_tokenize(str):
#         if w not in stop_words and w.isalnum():
#             ret_str = ret_str + " " + ps.stem(w)
#     return ret_str.strip()

# Stems words to their root words and removes all characters that are not alphabets
def stem_str(str):
    ret_str = ""
    for w in word_tokenize(str.lower()):
        if w not in stop_words and w.isalpha() and len(w) > 1:
            ret_str = ret_str + " " + ps.stem(w)
    return ret_str.strip()

# Gets the count of most frequent words give a dataframe
def word_freq(df):
    word_frequency = {}
    word_frequency_lst = []
    for index,row in df.iterrows():
        for w in word_tokenize(row['stemmed_sms']):
            if w not in word_frequency:
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1

    for key, value in word_frequency.items():
        temp = [key, value]
        word_frequency_lst.append(temp)
    word_freq_df = pd.DataFrame(word_frequency_lst, columns=["Unique Words", 'Frequency'])
    word_freq_df = word_freq_df.sort_values(['Frequency'], ascending=False)
    return word_freq_df

if __name__ == '__main__':

    #Reading in the file via csv library
    filepath = 'C:\\Users\\Joash\\Desktop\\University Stuff\\4B uni stuff\\SYDE 522\\522 Project\\SMS_spam_or_ham\\spam'
    csvfile = open(filepath + '.csv', "rt")
    reader = csv.reader(csvfile)
    count = 0
    data =[]
    for row in reader:
        data.append(row)
        count+=1
    data = data[1:]
    df = pd.DataFrame(data, columns=['class', 'sms'])

    #stemming and the removal of stop words via stem_str() function
    df['stemmed_sms'] = df.loc[:,'sms'].apply(lambda x: stem_str(str(x)))

    #Adding a length column to the dataframe
    df['len'] = df.loc[:,'sms'].apply(lambda x: len(x))

    # Printing out the stemmed words to csv
    df.to_csv(filepath + '_result.csv', index=False)

    # Everything from here on out is EDA
    #getting the most frequent spam unique words
    spam_df = df[df['class'] == 'spam']
    # print(spam_df)
    spam_word_freq = word_freq(spam_df)
    print("Top 10 most occuring spam unique words are:")
    print(spam_word_freq[0:9])
    ## Prints out the word frequency of spam words to csv
    # spam_word_freq.to_csv(filepath + '_freq.csv', index=False)


    #getting the most frequent ham unique words
    ham_df = df[df['class'] == 'ham']
    # print(ham_df)
    ham_word_freq = word_freq(ham_df)

    ## Prints out the word frequency of ham words to csv
    # ham_word_freq.to_csv(filepath + '_freq.csv', index=False)

    print("Top 10 most occuring ham unique words are:")
    print(ham_word_freq[0:9])

    #getting the most frequent unique words (ham or spam)
    word_freq = word_freq(df)
    ## Prints out the word frequency of words to csv
    # word_freq.to_csv(filepath + '_freq.csv', index=False)

    print("Top 10 most occuring unique words are in the whole dataset are:")
    print(word_freq[0:9])

    # Getting the data class labels and their counts
    count_spam =df[df['class'] == 'spam']['class'].count()
    print()
    print('Count of spam class is',count_spam)
    count_ham =df[df['class'] == 'ham']['class'].count()
    print()
    print('Count of ham class is',count_ham)
