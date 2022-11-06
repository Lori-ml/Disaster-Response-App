from sqlalchemy import create_engine
warnings.filterwarnings("ignore") 
import pandas as pd
import warnings
import sys


def load_data(messages_filepath, categories_filepath):
    
    '''Input:  Messages and categories csv files.
            
       Output: Merged dataset. '''
    
    #Load messages dataset
    messages = pd.read_csv(str(messages_filepath))
    #Load categories dataset
    categories = pd.read_csv(str(categories_filepath))
    #Merge datasets
    df = pd.merge(messages , categories , how ="inner" , on="id")
    
    return df


def clean_data(df):
    
    '''Input: Merged dataframe from load_data function
       
       output: Returns dataframe transformed and cleaned.  '''
    
    #Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';' , expand = True)
   
    #Select the first row of the categories dataframe
    row = categories[:1].values.flatten().tolist()
    #Use this row to extract a list of new column names for categories. 
    #Up to the second to last character of each string with slicing
    category_colnames = [words[:-2] for words in row]
    
    #Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        #Set each value to be the last character of the string
        categories[column]  = categories[column].astype(str).str[-1:]

        #Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Drop the original categories column from `df`
    df.drop(columns = 'categories' , inplace = True)
    
    #Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([categories , df] , axis = 1, join = 'inner')
    
    #Drop duplicates
    df.drop_duplicates(inplace = True)
   
    return df


def save_data(df, database_filename):
    '''Input: Cleaned dataframe returned from clea_data function , SQLite database filename
       Output: Creates table DisasterResponse where it inserts the data
    '''
    
    engine = create_engine('sqlite:///'+ str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False ,  if_exists='replace')  


def main():
    '''Starts the execution of above functions'''
     
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()