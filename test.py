from flask import Flask, render_template, request
# Load libraries

import pandas as pd
import numpy as np
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



app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])

def index():
    if request.method == 'POST':
        search = request.form['nm']
        df = pd.read_csv('./static/London_hotel_reviews.csv', encoding='latin-1')
        df = df.loc[df['ReviewText'].str.contains('foo') == True]

        # remove punctuation from all DOCs
        exclude = set(string.punctuation)
        alldocslist = []

        for index, i in enumerate(search):
            text = search
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
        for x in range(len(search)):
            lowers = [word.lower() for word in plot_data[0][x]]
            plot_data[0][x] = lowers

        print(plot_data[0][1][0:4])

        # remove stop words from all docs
        stop_words = set(stopwords.words('english'))

        for x in range(len(search)):
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
        print(stem1)
        print(stem2)

        less_data = df[df['ReviewText'].str.contains('|'.join(stem2))]
        less_data= less_data[['ReviewText', 'Location Of The Reviewer']][0:20]
        less_data = less_data.sort_values('Location Of The Reviewer', ascending=False)
        less_data = less_data.values.tolist()
        return render_template("result.html", tables=less_data, stemmer = stem2)
    return render_template("index.html")

app.run()
