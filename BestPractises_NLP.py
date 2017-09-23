
#Load Dependencies
import nltk
from nltk import word_tokenize, sent_tokenize
import gensim
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.io import output_notebook, output_file
from bokeh.plotting import show, figure
%matplotlib inline

nltk.download('punkt')

# new!
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *
from gensim.models.phrases import Phraser, Phrases
from keras.preprocessing.text import one_hot

nltk.download('stopwords')

#Load Data
nltk.download('gutenberg')
from nltk.corpus import gutenberg

#iteratively preprocess a sentence
#tokenised sentence
gberg_sents[4]
#to lowercase
[w.lower() for w in gberg_sents[4]]
#remove stopwords and punctuation 
stpwrds = stopwords.words('english') + list(string.punctuation)
stpwrds
[w.lower() for w in gberg_sents[4] if w not in stpwrds]
#stemwords
stemmer = PorterStemmer()
[stemmer.stem(w.lower()) for w in gberg_sents[4] if w not in stpwrds]
#handle bigram collocations
phrases = Phrases(gberg_sents) # train detector
bigram = Phraser(phrases) # create a more efficient Phraser object for transforming sentences
bigram.phrasegrams # output count and score of each bigram
"Jon lives in New York City".split()



