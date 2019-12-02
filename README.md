<h1>Text Search Based on TF-IDF Score</h1>

In order to perform a very similar search method to a simple Google edition, the simplest way is to use the Bag of Words model to measure each word in the dataset. Therefore, when you enter a query, the application will return the documents with the largest total weight in the query terms. I will be using Python-Flask in this tutorial. You can flow with me together.


<b>PART I: Preparing the documents/webpages</b>

We begin to work in the backend at first. We have to load important libraries for this purpose. We need numpy libraries, pandas and nltk for this project. We will then use the csv file open command. We must use the pandas library for this purpose. I take a list of hotel feedback I find in Kaggle for this project. 

          # Load libraries
          import pandas as pd
          import numpy as np
          import nltk
          df = pd.read_csv('./static/fewLondon.csv', encoding='latin-1')
          df = df.loc[df['ReviewText'].str.contains('foo') == False]
          
Now we have to remove punctuation from all documents and tokenize words in all documents. When this is done, we need to make all the words lowercase for all documents. There are many terms that are not so important to consider because they are used many times in every document such as pronouns, to-be verbs, or prepositions. Such words are known as stop words. The next step is to remove stop words from all documents. Moreover, one word has many various forms in English, such as tenses or plural. So, we also need to perform stemming to shorten these words and treat them as the same. After these 2 steps, we cut down a big size of memory to store all the terms.

       # remove punctuation from all DOCs
        exclude = set(string.punctuation)
        alldocslist = []

        for index, i in enumerate(searching):
            text = searching
            text = ''.join(ch for ch in text if ch not in exclude)
            alldocslist.append(text)
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
          
<b>PART II: CREATING THE INVERSE-INDEX</b>

Now, we need to create inverse index which gives document number for each document and where word appears. At first, we need to create a list of all words. Then we have to make one-time indexing which is the most processor-intensive step and will take time to run. But we only need to run this once because we are going to save this to avoid recalculation.

          TF (Term Frequency) measures the frequency of a word in a document.
          
          TF = (Number of time the word occurs in the text) / (Total number of words in text)
          
          IDF (Inverse Document Frequency) measures the rank of the specific word for its relevancy within the text. 
          
          IDF = (Total number of documents / Number of documents with word t in it)
          
          Thus, the TF-IDF is the product of TF and IDF:
          TF-IDF = TF * IDF


          tfidf_paragraph = tfidf(df['ReviewText'])
          
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
                        
<b>PART III: Setting up Frontend section and Finalizing the web application</b>

We then create word search which takes multiple words and finds documents that contain both along with metrics for ranking. The results are sorted according to TF-IDF score. We use HTML5, CSS3, bootstrap 4 and javascript for the front end and we use javascript to highlight search terms. When the application is ready, it should work like this http://jeewangw.pythonanywhere.com/

<b>Contribution and Challenging</b>

The dataset that I found on the kaggle was not correctly formatted and encoded. So, it created many problems while working on backend. I spent many hours trying to fix the problem. At first, I thought to change the dataset, later I decided to fix the problems and did more research on the problem. Finally, after a week, I came to the solution myself. I had to use try-catch exception in many sections of the code to make it workable. It was the most challenging part while working on Text Search. One of my contributions was fixing that problem. Other contributions were saving calculated TF-IDF score in CSV format for each document and designing and developing the front-end. Most of the back-end part was already done but it was not made for my dataset. So, I had to modify 60% of it. 

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
          
<b>Experiments</b>

The 'english' stemmer is better than the original 'porter' stemmer.

          print(SnowballStemmer("english").stem("generously"))
           generous
          print(SnowballStemmer("porter").stem("generously")) 
          gener
          
          
          # stem words (could try others/lemmers )

          snowball_stemmer = SnowballStemmer("english")
          stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
          stem1 = stemmed_sentence

          porter_stemmer = PorterStemmer()
          stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
          stem2 = stemmed_sentence
<b>References</b>

Dataset is taken from: https://www.kaggle.com/PromptCloudHQ/reviews-of-londonbased-hotels
Formulae for TF-IDF is taken from Amit Kumar Jaiswal article: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine



<h1>Image Caption Search</h1>

People are often confused with Image Captioning and Image Classification. Image Classification tells which object, like dog, cat or people, is in the picture. While Image Captioning gives a description of the image like a dog is having food. In this article, I give you an idea of how to build Image Captioning and search those captions based on the TFIDF score.

Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given image. It requires both computer vision concept to process the image and model of NLP to understand the image and right it in words I use the Tensor flow official website for understanding the code and use fliker dataset to train the model.

Libraries of python 3 which I used are Matplotlib, Pandas, numpy, and Keras.

I built my image captioning model by taking guideline from this site: https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

Before starting building Image Captioning, make sure you setup your environment for machine learning. Here is the site that I took reference to setup my python/anaconda environment: https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

If you follow the steps carefully without missing any part, you will get results in 3/4 hours depending upon the speed of your computer. Captions are generated in a text file so I converted it in Excel file. Now, it's time to build a search engine that searches the user's searched caption in the dataset and retrieves results based on the sorting order of the TFIDF result. I already have created Text Search in Phase 1. The algorithm is the same. Here is the link to my Phase 1 documentation: https://medium.com/@jeevangyawali_35083/text-search-based-on-tf-idf-score-a0a0ceff1cad

<b>Contribution and Challenging</b>

I modified the code to work for my image dataset and exported the results in xlms format. Hosting 1.03 GB images was challenging. I used 000webhost to host 1 GB images in free of cost. Out of 8k images, 500 images were not uploaded in the hosting site because of the size limit. The web application doesn't display those 500 images when a user searches on the application. Here is the link where I uploaded flickr8k images: http://jeevangyawaliflicker8kimages.000webhostapp.com/?dir=Images/

<b>References</b>

https://www.tensorflow.org/tutorials/text/image_captioning
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/
https://medium.com/@jeevangyawali_35083/text-search-based-on-tf-idf-score-a0a0ceff1cad
http://jeevangyawaliflicker8kimages.000webhostapp.com/?dir=Images/
