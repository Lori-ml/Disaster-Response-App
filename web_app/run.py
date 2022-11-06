import pandas as pd
from Tokenize import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import render_template, request, jsonify
from sqlalchemy import create_engine
from flask import Flask
from plotly.graph_objs import Bar
import joblib
import pickle
import plotly
import json


app = Flask(__name__)


#Load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
         
df = pd.read_sql_table('DisasterResponse', engine)

#Load model
model = joblib.load("models/classifier.pkl")


#Index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #Count messages for each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    df_categories = df[['related', 'request', 'offer', 'aid_related', 'medical_help',
       'medical_products', 'search_and_rescue', 'security', 'military',
       'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
       'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report']]

    #Get the sum for top 10 categories
    df_top_10_catg = df_categories.sum().sort_values(ascending = False).head(10).to_frame().reset_index().rename(columns={'index' :'Category', 0:'Count'})
    
    #Get the sum for bottom 10 categories
    df_tail_10_catg = df_categories.sum().sort_values(ascending = False).tail(10).to_frame().reset_index().rename(columns={'index' :'Category', 0:'Count'})
    
    
    #Visuals
    graphs = [
        #Genres chart
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
         ,
        
        #Categories Chart Top 10 
        
         {
            'data': [
                Bar(
                    x=df_top_10_catg['Category'],
                    y=df_top_10_catg['Count']
                )
            ],

            'layout': {
                'title': 'Distribution of Top 10 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        } ,
        
        #Categories Chart Bottom 10 
        
         {
            'data': [
                Bar(
                    x=df_tail_10_catg['Category'],
                    y=df_tail_10_catg['Count']
                )
            ],

            'layout': {
                'title': 'Distribution of Bottom 10 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON  )


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df[df.columns[~df.columns.isin(['message' , 'id' , 'original' ,'genre'])]], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )




def main():
    app.run()
    

if __name__ == '__main__':
   main()