## Results 

Below images show results from a test run in the application.

![Main Page](https://github.com/Lori-ml/Disaster-Response-App/blob/main/images/Main%20page.PNG)

![Prediction](https://github.com/Lori-ml/Disaster-Response-App/blob/main/images/Classification.PNG)

## [Link to the app](https://dashboard.heroku.com/apps/myapp-worldbankdata-florida-1)

## Project Description

Aim of this project is to help emergency workers classify disaster related messages in several categories. Data has been provided by [Appen](https://appen.com/) and contains over 26 thousand real messages sent during disaster events. Messages entered are classified in several categories with an accuracy of 95%. 

## Local run

Following commands need to be run in project's root directory

1. Run ETL pipeline that cleans data and stores them in a SQLite database.
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
2. Run ML pipeline that trains the model and saves it as a pickle file
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
3. Run the file that renders the application to a webpage.
```bash
python3 run.py
```

## File Description 

data 
 - process_data.py - ETL pipeline that cleans data and stores to DisasterReponse.db

models
 - train_classifier.py - ML pipeline that trains the model and saves it as a pickle file
 - Tokenize.py - Serves as a library for train_classifier.py for tokenizing and cleaning text data.

Jupyter Notebooks

- ETL Pipeline Preparation - Prepares and explores datasets, functions created are later encapsulated in  process_data.py ETL pipeline

- ML Pipeline Preparation -  Loads data created from above notebook, cleans message text data, tests different models and prints f1 score, precision and recall for each category in the test dataset. Functions created along with the best model are later encapsulated in train_classifier.py

run.py - This script is called by flask to load the application in web

Procfile - Is needed if application will be deployed in a server

requirements.txt - Contains all necessary packages for this application, is needed if the app will be deployed.

nltk.txt - Nltk libraries, is needed if the app will be deployed.

## Acknowledgements
Data used in this project was provided by [Appen](https://appen.com/).
