# PART I: Preparing the documents/webpages
When you want to search for something online, you look onto Google or Bing. They contain more data and also they are faster as their algorithm is really good. In this article, I am going to implement a search feature which is very similar to a smaller version of Google or Bing search engine. I will be using Python-Flask in this tutorial. We can flow with me together.
 
At first we start to work in backend. For that we need to load essential libraries. For this project, we need numpy, pandas and nltk libraries.
 
After that, we need to use command to open csv file. For this we have to use pandas library. For this project, I am taking dataset of Hotel Reviews which I have found in Kaggle. Now we have to remove punctuation from all documents and tokenize words in all documents. When this is done, we need to make all words lowercase for all documents.There are many terms that are not so important to consider, because they are used many times in every document such as pronouns, to-be verbs, or prepositions. Such words are known as stop words. Next step is to remove stop words from all documents. Moreover, in English, one words usually have many different forms such as tenses or plural. So, we also need to perform stemming to shorten these words and treat them as the same. After these 2 steps, we cut down a big size of memory to store all the terms.
 
# PART II: CREATING THE INVERSE-INDEX
Now, we need to create inverse index which gives document number for each document and where word appears. At first, we need to create a list of all words. Then we have to make  one-time indexing which is  the most processor-intensive step and will take time to run. But we only need to run this once because we are going to save this to avoid re calculation.
 
# PART III: The Search Engine
We then create word search which takes multiple words and finds documents that contain both along with metrics for ranking:
 
    ## (1) Number of occruances of search words 
    ## (2) TD-IDF score for search words 
    ## (3) Percentage of search terms
    ## (4) Word ordering score 
    ## (5) Exact match bonus
    ## (6) Rank and Return
 
 
 # Refrence
 Dataset is taken from: https://www.kaggle.com/PromptCloudHQ/reviews-of-londonbased-hotels
 
 Formulae for TF-IDF is taken from Amit Kumar Jaiswal article: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine
 
 
