import streamlit as st
import pandas as pd
import numpy as np
st.title('Sentiment Analysis Arcana Hackathon')
from coccurence import *

#streamlit run main.py

option = st.selectbox('Choose an stock index', ['AAPL', 'NVDA', 'TSLA', 'AMD', 'JNUG', 'JDST', 'LABU', 'QCOM', 'INTC', 'DGAZ'])

comp= []
comp.append(option)
get_json(tweets_per_symbol = 3000)
cooccurence(option)


#sentiment analysis
#get_json(comp, tweets_per_symbol = 3000, path1 = './Testdata/')
sentimentanalysis(option)