#Import libraries
from nltk.tokenize import word_tokenize
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.exceptions import UndefinedMetricWarning
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nltk
import re
def warn(*args, **kwargs):
    pass
import warnings
warnings.filterwarnings('always')
warnings.warn = warn

warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 

nltk.download(['punkt', 'wordnet', 'stopwords' ])



def tokenize(text):
    '''Input: text data
       Output: text data cleaned
    '''
    #Convert to lower cases
    text = text.lower()
    #Remove punctuation 
    text = re.sub(r'[^\w\s]', '', text)

    # Loop all words and replace with empty space to remove numbers 
    for i in text:
        if i.isdigit():
            text = text.replace(i, '')

    #Tokenzie      
    words = word_tokenize(text)
 
    #Remove stopwords

    words = [w for w in words if w not in stopwords.words("english")]

    #Reduce words to their root form , overwrite the default parameter 'words' to 'verb'
    lemmed = [WordNetLemmatizer().lemmatize(w , pos = 'v') for w in words]

    return lemmed