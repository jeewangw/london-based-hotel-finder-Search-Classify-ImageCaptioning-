import json

from flask import Flask, render_template, request
# Load libraries
import pandas as pd
import numpy as np
import re
import string
import random
import nltk
from nltk.corpus import brown
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import math
from textblob import TextBlob as tb
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True

def tfidf( paragraph ):
    # Tokenize using the white spaces
    dictOfWords = {}
    for index, sentence in enumerate(paragraph):
        try:
            tokenizedWords = nltk.tokenize.WhitespaceTokenizer().tokenize(paragraph[index])
            dictOfWords[index] = [(word, tokenizedWords.count(word)) for word in tokenizedWords]
        except KeyError:
            continue

    # second: remove duplicates
    termFrequency = {}

    for i in range(0, len(paragraph)):
        listOfNoDuplicates = []
        try:
            for wordFreq in dictOfWords[i]:
                if wordFreq not in listOfNoDuplicates:
                    listOfNoDuplicates.append(wordFreq)
                termFrequency[i] = listOfNoDuplicates
        except KeyError:
            continue

    # Third: normalized term frequency
    normalizedTermFrequency = {}
    for i in range(0, len(paragraph)):
        try:
            sentence = dictOfWords[i]
            lenOfSentence = len(sentence)
            listOfNormalized = []
            for wordFreq in termFrequency[i]:
                normalizedFreq = wordFreq[1] / lenOfSentence
                listOfNormalized.append((wordFreq[0], normalizedFreq))
            normalizedTermFrequency[i] = listOfNormalized
        except KeyError:
            continue

    # print(normalizedTermFrequency)
    # ---Calculate IDF

    # First: put al sentences together and tokenze words

    allDocuments = ''
    for sentence in paragraph:
        allDocuments += sentence + ' '
    print (allDocuments)
    allDocumentsTokenized = allDocuments.split(' ')

    # print(allDocumentsTokenized)
    allDocumentsNoDuplicates = []

    for word in allDocumentsTokenized:
        if word not in allDocumentsNoDuplicates:
            allDocumentsNoDuplicates.append(word)

    # print(allDocumentsNoDuplicates)
    # Second calculate the number of documents where the term t appears

    dictOfNumberOfDocumentsWithTermInside = {}

    for index, voc in enumerate(allDocumentsNoDuplicates):
        count = 0
        for sentence in paragraph:
            if voc in sentence:
                count += 1
        dictOfNumberOfDocumentsWithTermInside[index] = (voc, count)

    # print(dictOfNumberOfDocumentsWithTermInside)

    # calculate IDF

    dictOFIDFNoDuplicates = {}

    for i in range(0, len(normalizedTermFrequency)):
        listOfIDFCalcs = []
        try:
            for word in normalizedTermFrequency[i]:
                for x in range(0, len(dictOfNumberOfDocumentsWithTermInside)):
                    if word[0] == dictOfNumberOfDocumentsWithTermInside[x][0]:
                        listOfIDFCalcs.append(
                            (word[0],
                             math.log(len(paragraph) / dictOfNumberOfDocumentsWithTermInside[x][1])))
            dictOFIDFNoDuplicates[i] = listOfIDFCalcs
        except KeyError:
            continue
    # print(dictOFIDFNoDuplicates)

    # Multiply tf by idf for tf-idf

    dictOFTF_IDF = {}
    for i in range(0, len(normalizedTermFrequency)):
        listOFTF_IDF = []
        try:
            TFsentence = normalizedTermFrequency[i]
            IDFsentence = dictOFIDFNoDuplicates[i]
            for x in range(0, len(TFsentence)):
                try:
                    listOFTF_IDF.append((TFsentence[x][0], TFsentence[x][1] * IDFsentence[x][1]))
                except IndexError:
                    continue


            dictOFTF_IDF[i] = listOFTF_IDF
        except KeyError:
            continue
    #print(dictOFTF_IDF)
    return dictOFTF_IDF

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])

def index():
    if request.method == 'POST':
        searching = request.form['nm']
        df = pd.read_csv('./static/fewLondon.csv', encoding='latin-1')
        df = df.loc[df['ReviewText'].str.contains('foo') == False]
        #df["new_column"] = df['ReviewText'].str.replace('[^\w\s]', '')
        #alldocs = df.apply(lambda row: nltk.word_tokenize(row['new_column']), axis=1)
        #print (alldocs)

        # or export it as a list of dicts
        #dicts = df.to_dict().values()

        # remove punctuation from all DOCs
        exclude = set(string.punctuation)
        alldocslist = []

        for index, i in enumerate(searching):
            text = searching
            text = ''.join(ch for ch in text if ch not in exclude)
            alldocslist.append(text)

        print(alldocslist[1])

        # tokenize words in all DOCS
        plot_data = [[]] * len(alldocslist)

        for doc in alldocslist:
            text = doc
            tokentext = word_tokenize(text)
            plot_data[index].append(tokentext)

        # make all words lower case for all docs
        for x in range(len(searching)):
            lowers = [word.lower() for word in plot_data[0][x]]
            plot_data[0][x] = lowers

        print(plot_data[0][1][0:4])

        # remove stop words from all docs
        stop_words = set(stopwords.words('english'))

        for x in range(len(searching)):
            filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
            plot_data[0][x] = filtered_sentence

        print(plot_data[0][1][0:4])

        # stem words EXAMPLE (could try others/lemmers )

        snowball_stemmer = SnowballStemmer("english")
        stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
        stem1 = stemmed_sentence

        porter_stemmer = PorterStemmer()
        snowball_stemmer = SnowballStemmer("english")
        stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
        stem2 = stemmed_sentence

        #tfidf_paragraph = tfidf (df['ReviewText'] [0:1000])
        #tfidf_paragraph = tfidf(df['ReviewText'])
        # pickel (save) the dictonary to avoid re-calculating

        #pickle.dump(tfidf_paragraph, open("worddic_1000.p", "wb"))
        tfidf_paragraph = pickle.load(open("worddic_1000.p", "rb"))
        tfidf_query = tfidf (stem2)

        print (tfidf_paragraph)
        print (tfidf_query)
        print (plot_data[0][1])
        sech = plot_data[0][1]

        aTuple = []
        for i in tfidf_query:
            for j in tfidf_query[i]:
                if (j[0] in stem2):
                    Tuple = (i , j[0], j[1])
                    aTuple.append (Tuple)
        aList = list(aTuple)
        print (aList)

        pTuple = []
        indexs = []
        values = []
        for i in tfidf_paragraph:
            for j in tfidf_paragraph[i]:
                if (j[0] in stem2):
                    if i not in indexs:
                        indexs.append(i)
                        values.append(j[1])
                        Tuple = (i, j[0], j[1])
                        pTuple.append(Tuple)
                    else:
                        b = j[1] + values[-1]
                        values.append(b)
        pList = list(pTuple)
        print(pList)
        print(values)
        print (indexs)
        # creating a blank series
        Type_new = pd.Series([])
        m = 0
        for index in range(len(df.index)):
            a = index
            if a in indexs:
                    Type_new[a] = values[m]
                    m = m+1
            else:
                Type_new[a] = 0

        # inserting new column with values of list made above
        df.insert(6, "Type New", Type_new)
        print (df["Type New"])

        less_data = df[df['ReviewText'].str.contains('|'.join(stem2))]
        less_data = less_data.sort_values('Type New', ascending=False)
        less_data= less_data[['Property Name', 'ReviewText','Type New']][0:20]

        less_data = less_data.values.tolist()
        #print(less_data)
        return render_template("result.html", tables=less_data, stemmer = stem2)
    return render_template("index.html")

app.run()
