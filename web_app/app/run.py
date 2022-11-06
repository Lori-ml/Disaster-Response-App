import json
import plotly
import pandas as pd
from Tokenize import tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)



#def tokenize(text):
 #   tokens = word_tokenize(text)
 #  lemmatizer = WordNetLemmatizer()

  #  clean_tokens = []
   # for tok in tokens:
    #    clean_tok = lemmatizer.lemmatize(tok).lower().strip()
     #   clean_tokens.append(clean_tok)

   # return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
         
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
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


    df_top_10_catg = df_categories.sum().sort_values(ascending = False).head(10).to_frame().reset_index().rename(columns={'index' :'Category', 0:'Count'})
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
        
        #Categories Chart Tail 10 
        
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
    app.run(debug = True)
    #app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
   main()