# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:10:22 2022

@author: descalante
"""

from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import re

df = pd.read_csv("/Users\descalante\.spyder-py3\clean_tickers_v1.csv")
df.columns = ['TICKER','WORDCOUNT','DESCRIPTION','BUSINESS_SUMMARY']




des = []
doc =[]
doc2 = []

for i in range (len(df)):
    url = 'https://finance.yahoo.com/quote/'+str(df.loc[i,"TICKER"])+'?p='+str(df.loc[i,"TICKER"])+'&.tsrc=fin-srch'
    response = get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    var = soup.select("p[class^=businessSummary]")
    var = str(var)
    pattern = re.compile('>(.*)<')
    for a in re.findall(pattern, var):
        a = str(a)
        stuff = a. lstrip(">"). rstrip("<") 
        df.loc[i,'BUSINESS_SUMMARY'] = stuff
        
df.to_csv("/Users\descalante\.spyder-py3\clean_tickers_des.csv")


'''
for stock in stock_data:
    print(stock.text)
    '''
    
     