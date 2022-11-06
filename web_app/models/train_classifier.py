# import libraries
from nltk.tokenize import word_tokenize
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from Tokenize import tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import nltk
import pickle
import warnings
import sys
import re
warnings.filterwarnings('always')


def load_data(database_filepath):
    '''Input: database filepath
       Output: X - dataset containing messeages 
               Y - dataset containing 36 different classifications 
               category names - 36 categories presented in a list
    '''
    #Load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table( 'DisasterResponse' , engine)
    #Save message data in X
    X = df['message']
    #Keep only categories for Y
    Y = df[df.columns[~df.columns.isin(['message' , 'id' , 'original' ,'genre'])]]
    #Create a list with categories names
    category_names = list(Y.columns)
    
    return X , Y , category_names



def build_model():
    
    #Create a pipeline for Random Forest
    pipeline_rf = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1))) 
    ])

    #Create parameters that will be used later in a grid search
    parameters = {
            'tfidf__use_idf': (True, False),
            'clf__estimator__n_estimators': [10,15]
            }

    #Find best parameters for RandomForest Classifier
    cv_rf = GridSearchCV(pipeline_rf, param_grid=parameters) 
    
    return cv_rf


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    for i , col_name in zip(range(len(category_names)) , category_names):
        print(f"Metrics for {col_name} :" )
        print(classification_report(Y_test[col_name], y_pred[: , i]))

    accuracy = (y_pred == Y_test.values).mean()

    print(f"Model accuracy is {accuracy:.2f}" )


def save_model(model, model_filepath):
    rf_model = str(model_filepath)
    with open (rf_model, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()










