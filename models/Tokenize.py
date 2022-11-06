#Import libraries
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
import pandas as pd
import re



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