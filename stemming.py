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
        if w not in stop_words and w.isalpha():
            ret_str = ret_str + " " + ps.stem(w)
    return ret_str.strip()

# Gets the count of most frequent words give a dataframe
def word_freq(df):
    word_frequency = {}
    for index,row in df.iterrows():
        for w in word_tokenize(row['stemmed_sms']):
            if w not in word_frequency:
                word_frequency[w] = 1
            else:
                word_frequency[w] += 1
    return word_frequency

if __name__ == '__main__':


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
    # print(df)
    df['stemmed_sms'] = df.loc[:,'sms'].apply(lambda x: stem_str(str(x)))
    # print(df)
    # df.to_csv(filepath + '_test3.csv', index=False)

    spam_df = df[df['class'] == 'spam']
    # print(spam_df)
    word_freq = word_freq(spam_df)