import sys

sys.path.append(r'c:\users\siddh\arcana\lib\site-packages')

import io, json, requests, time, os, os.path, math, urllib
import streamlit as st
from sys import stdout
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from pandas_datareader.data import get_data_yahoo
import pandas_datareader.data as pdr
import yfinance as yfin
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import requests


def get_response(symbol, older_than, retries=5):
    url = 'https://api.stocktwits.com/api/2/streams/symbol/%s.json?max=%d' % (symbol, older_than - 1)
    for _ in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return json.loads(response.content)
        elif response.status_code == 429:
            print(response.content)
            return None
        # time.sleep(1.0)
    # couldn't get response
    return None


def get_older_tweets(symbol, num_queries, path):
    # path = './data/%s.json' % symbol
    if os.path.exists(path):
        # extending an existing json file
        with open(path, 'r') as f:
            data = json.load(f)
            if len(data) > 0:
                older_than = data[-1]['id']
            else:
                older_than = 1000000000000
    else:
        # creating a new json file
        data = []
        older_than = 1000000000000  # any huge number

    for i in range(num_queries):
        content = get_response(symbol, older_than)
        if content == None:
            print('Error, an API query timed out')
            break
        data.extend(content['messages'])
        older_than = data[-1]['id']
        stdout.write('\rSuccessfully made query %d' % (i + 1))
        stdout.flush()
        # sleep to make sure we don't get throttled
        # time.sleep(0.5)

    # write the new data to the JSON file
    with open(path, 'w') as f:
        json.dump(data, f)
    # print
    print('Done')


def get_json(symbols=['AAPL', 'NVDA', 'TSLA', 'AMD', 'JNUG', 'JDST', 'LABU', 'QCOM', 'INTC', 'DGAZ'],
             tweets_per_symbol=3000, path1='./data/'):
    if not os.path.exists(path1):
        os.mkdir(path1)

    for symbol in symbols:
        path = path1 + symbol + '.json'
        # print(path1, path)
        if os.path.exists(path):
            with open(path, 'r') as f:
                num_tweets = len(json.load(f))
        else:
            num_tweets = 0
        num_queries = (int(tweets_per_symbol) - int(num_tweets) - 1) // 30 + 1
        if num_queries > 0:
            print('Getting tweets for symbol %s' % symbol)
            get_older_tweets(symbol, num_queries, path=path)


# Function takes in a JSON and returns a Pandas DataFrame for easier operation.
def stocktwits_json_to_df(data, verbose=False):
    # data = json.loads(results)
    columns = ['id', 'created_at', 'username', 'name', 'user_id', 'body', 'basic_sentiment', 'reshare_count']
    db = pd.DataFrame(index=range(len(data)), columns=columns)
    for i, message in enumerate(data):
        db.loc[i, 'id'] = message['id']
        db.loc[i, 'created_at'] = message['created_at']
        db.loc[i, 'username'] = message['user']['username']
        db.loc[i, 'name'] = message['user']['name']
        db.loc[i, 'user_id'] = message['user']['id']
        db.loc[i, 'body'] = message['body']
        # We'll classify bullish as +1 and bearish as -1 to make it ready for classification training
        try:
            if (message['entities']['sentiment']['basic'] == 'Bullish'):
                db.loc[i, 'basic_sentiment'] = 1
            elif (message['entities']['sentiment']['basic'] == 'Bearish'):
                db.loc[i, 'basic_sentiment'] = -1
            else:
                db.loc[i, 'basic_sentiment'] = 0
        except:
            db.loc[i, 'basic_sentiment'] = 0
        # db.loc[i,'reshare_count'] = message['reshares']['reshared_count']
        for j, symbol in enumerate(message['symbols']):
            db.loc[i, 'symbol' + str(j)] = symbol['symbol']
        if verbose:
            # print message
            print(db.loc[i, :])
    db['created_at'] = pd.to_datetime(db['created_at'])
    return db


def cooccurence(symbol='INTC', numb=10):
    path = './data/' + symbol + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    db = stocktwits_json_to_df(data)
    print('%d examples extracted ' % db.shape[0])

    enddate = db['created_at'].max()
    startdate = db['created_at'].min()

    def countcomentions(df):
        def getsymbolset(df):
            symbols = []
            for i, row in df.iterrows():
                for symbol in row:
                    if (pd.notnull(symbol)):
                        symbols.append(symbol)
            return set(symbols)

        def getallsymbols(df):
            columns = df.columns
            symbolcolumns = []
            for col in columns:
                if col.startswith('symbol'):
                    symbolcolumns.append(col)
            return df[symbolcolumns]

        def count(df, stock_symbol):
            cnt = Counter()
            for i, row in df.iterrows():
                for sym in row:
                    if (sym != stock_symbol) & pd.notnull(sym):
                        cnt[sym] += 1
            return cnt

        df = getallsymbols(df)
        symbolset = getsymbolset(df)
        print(len(symbolset), "total symbols found.")
        co = np.zeros((len(symbolset), len(symbolset)))
        co = pd.DataFrame(co, index=list(symbolset), columns=list(symbolset))
        for i, row in df.iterrows():
            for stock_symbol in row:
                for sym in row:
                    if (sym != stock_symbol) & pd.notnull(stock_symbol) & pd.notnull(sym):
                        co.loc[stock_symbol, sym] += 1
        return co
        # return pd.DataFrame(co)

    co = countcomentions(db)
    # st.title('FFFFF')
    coocc = co.loc[symbol, co.loc[symbol, :] > 0].sort_values(ascending=False)[:numb]
    column_names = coocc.index.tolist()
    # print(coocc )
    st.subheader(' Most Mentioned Stocks on tweets on %s' % symbol)

    st.bar_chart(coocc)


def sentimentanalysis(symbol='GOOGL'):
    def isSentiment(data):
        datareturn = []
        for i in range(len(data)):
            if (data[i]['entities']['sentiment'] != None):
                datareturn.append(data[i])
        return datareturn

    def get_tweets_and_labels(data):
        # filter out messages without a bullish/bearish tag
        data = isSentiment(data)
        # get tweets
        tweets = map(lambda m: m['body'], data)

        # get labels

        def create_label(message):
            sentiment = message['entities']['sentiment']['basic']
            if sentiment == 'Bearish':
                return 0
            elif sentiment == 'Bullish':
                return 1
            else:
                raise Exception('Got unexpected sentiment')

        labels = map(create_label, data)
        return tweets, labels

    tweets = []
    labels = []
    all_tweets = []
    for filename in os.listdir('./data'):
        path = './data/%s' % filename
        if filename != symbol + '.json':
            print(path)
            with open(path, 'r') as f:
                data = json.load(f)
            all_tweets.extend(map(lambda m: m['body'], data))
            t, l = get_tweets_and_labels(data)
            tweets.extend(t)
            labels.extend(l)

    assert (len(tweets) == len(labels))

    # print('%d labeled examples extracted ' % len(tweets))

    def tfidf_vectorizer(tweets, all_tweets=None):
        vectorizer = TfidfVectorizer()
        if all_tweets != None:
            # use all tweets, including unlabeled, to learn vocab and tfidf weights
            vectorizer.fit(all_tweets)
        else:
            vectorizer.fit(tweets)
        return vectorizer

    def train_svm(X, y):
        model = svm.LinearSVC(penalty='l2', loss='hinge', C=1.0)
        # model = svm.SVC(C=1.0, kernel='rbf')
        model.fit(X, y)
        return model

    vectorizer = tfidf_vectorizer(tweets, all_tweets)
    X = vectorizer.transform(tweets)
    words = vectorizer.get_feature_names_out()
    y = np.array(labels)
    # print (X.shape)
    # print (y.shape)

    N = X.shape[0]
    num_train = int(math.floor(N * 0.9))
    P = np.random.permutation(N)
    X_tr = X[P[:num_train]]
    y_tr = y[P[:num_train]]
    X_te = X[P[num_train:]]
    y_te = y[P[num_train:]]
    # print ('Training set size is %d' % num_train)
    # print ('Percent bullish = %f%%' % (100*y.mean()))
    model = train_svm(X_tr, y_tr)
    # print ('Training set accuracy = %f' % model.score(X_tr, y_tr))
    # print ('Test set accuracy = %f' % model.score(X_te, y_te))
    model = linear_model.LogisticRegression(penalty='l2', C=1.0, class_weight='balanced')
    # model = svm.LinearSVC(penalty='l2', loss='hinge', C=1.0, class_weight='balanced')
    # from sklearn.svm import SVC
    # from sklearn.model_selection import StratifiedShuffleSplit
    # from sklearn.model_selection import GridSearchCV

    # C_range = np.logspace(-2, 2, 5)
    # gamma_range = np.logspace(-2, 2, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    # model = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    model.fit(X, y)

    # with open('./tsla_data/TSLA.json', 'r') as f:

    with open('./data/%s.json' % symbol, 'r') as f:
        data = json.load(f)[::-1]

    def extract_body(m):
        return m['body']

    def extract_date(m):
        return m['created_at']

    def extract_sentiment(m):
        if m['entities']['sentiment'] != None:
            sentiment = m['entities']['sentiment']['basic']
            if sentiment == 'Bearish':
                return 0
            else:
                return 1
        else:
            return np.nan

    d = {'body': map(extract_body, data),
         'date': pd.to_datetime(list(map(extract_date, data))),
         'sentiment': map(extract_sentiment, data)}
    df = pd.DataFrame(data=d)

    # use classifier to predict sentiment for unlabeled examples
    features = vectorizer.transform(df['body'])
    predictions = model.predict(features)
    predicted_sentiment = []
    for i, sentiment in enumerate(df['sentiment']):
        if np.isnan(sentiment):
            predicted_sentiment.append(predictions[i])
        else:
            predicted_sentiment.append(sentiment)
    df['predicted_sentiment'] = pd.Series(predictions)

    print(df.dtypes)
    print(df.head())

    grouped_df = df.groupby(pd.Grouper(key='date', freq='1D')).aggregate(np.mean)

    # print (grouped_df.head())
    st.subheader(' Predicted sentiment vs Actual sentiment ')
    st.line_chart(grouped_df[['sentiment', 'predicted_sentiment']])
    # st.line_chart(grouped_df[)
    plt.legend(loc='lower right')
    plt.xticks(rotation=45)
    grouped_df.dropna()
    print(grouped_df.shape)
    coef = np.corrcoef(grouped_df['sentiment'], grouped_df['predicted_sentiment'])[0, 1]
    # print
    st.write('Correlation coefficient = %f' % coef)