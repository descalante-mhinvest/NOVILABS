# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a script file.

"""

#imports

import requests
import edgar
import pandas as pd
from bs4 import BeautifulSoup
'''import matplotlib.pyplot as plt'''
import yfinance as yf


#mhinvest headers

HEADERS = {'User-Agent': 'Miller Howard Investments descalante@mhinvest.com'}

#download and read the edgar index, get rid of non-10k items
#using QTR1 file

edgar.download_index('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3', 2021,'descalante@mhinvest.com',skip_all_present_except_last=False)
df = pd.read_csv("U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/2023-QTR1.tsv", sep="|")
df.columns =['CIK', 'Company_Name', 'Form_Type', 'Filing_Date', 'TXT_URL', 'HTML_URL']
df = df.drop(df[df.Form_Type != "10-K"].index)


#I think our tickers are the names already in the clean energy menu
'''
df4 = pd.read_csv('U:/03_InvestmentTeamMembers/12_DAE/Anaconda3/our_tickers.csv')
df4.columns =['ticker', 'market_cap', 'ADVT', 'Is_Liquid']
df4 = df4.drop(df4[df4['Is_Liquid'] != "YES"].index)
'''
#GET CIK to tickers data

tickers_cik = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS)
tickers_cik = pd.json_normalize(pd.json_normalize(tickers_cik.json(),\
max_level=0).values[0])
tickers_cik["cik_str"] = tickers_cik["cik_str"].astype(str).str.zfill(10) #adjustment to the CIK numbers
#tickers_cik.set_index("ticker",inplace=True)

#Merge liquidity file with with edgar to get matching index (CIK Number)

df3 = pd.merge(df2, 
                      tickers_cik, 
                      on ='ticker', 
                      how ='inner')

#Merge clean energy menu tickers file with with edgar to get matching index (CIK Number)

df4 = pd.merge(df4, 
                      tickers_cik, 
                      on ='ticker', 
                      how ='inner')

#rename and format CIK columns

df3.rename(columns = {'cik_str':'CIK'}, inplace = True)

df['CIK'] = df['CIK'].apply(lambda x: '{0:0>10}'.format(x))
df['CIK'] = df['CIK'].astype(str)
tickers_cik.rename(columns = {'cik_str':'CIK'}, inplace = True)
#FIlter main data by liquidity

df = pd.merge(df, 
                      tickers_cik, 
                      on ='CIK', 
                      how ='inner')

#df = df.head(500)
#attach sector data to df

sectors = []

for i in range (len(df)):
    try:
        ticker = yf.Ticker(df.loc[i,"ticker"])
        sector = ticker.info["sector"]
        sectors.append(sector)
    except:
        sectors.append("not found")

df["sector"] = sectors

#keep Consumer Discretionary, Industrials, Materials, Info-tech, Utilities

keep_sectors = ["Consumer Cyclical", "Industrials", "Basic Materials", "Technology", "not found"]

df1 = df[df['sector'].isin(keep_sectors)]

df1.to_csv('U:/03_InvestmentTeamMembers/12_DAE/Anaconda3/LiquidityFileNew.csv')

#filtering names for liquidity

df2 = pd.read_csv('U:/03_InvestmentTeamMembers/12_DAE/Anaconda3/LiquidityFileNew2.csv')
#df2['index'] = range(1, len(df2) + 1)
#df2['index'] = df2.index
df2.columns =['CIK', 'Company_Name', 'Form_Type', 'Filing_Date', 'TXT_URL', 'HTML_URL','ticker','title','sector','Eikon RIC','Market Cap','ADVT','Liquidity']
df2['Liquidity'] = df2['Liquidity'].astype(str)
df2['HTML_URL'] = df2['HTML_URL'].astype(str)
df3 = df2.drop(df2[df2.Liquidity != "True"].index)
#df3['index'] = range(1, len(df3))
#df3['index'] = df3.index
df3 = df3.reset_index()
df3 = df3.drop(['index'], axis=1)


# Insert url address into html data column

def prepend(addy):
    df3['HTML_URL'] = addy + df3['HTML_URL']
    return(df3['HTML_URL'])

addy = 'https://www.sec.gov/Archives/'
prepend(addy)

#threshhold is constant for all 

wordcount = []
match = []
threshhold = 45

#Keywords list, not case sensitive

keywords = [
    'Renewable',
    'Wind',
    'Solar',
    'Hydroelectric',
    'Sustainable',
    'Electric Vehicle',
    'Ev Charging',
    'Charging Station',
    'Fuel-Cell',
    'fuel cell',
    'Carbon Capture',
    'Photovoltaic',
    'used oil',
    'hydraulic',
    'Thermal',
    'Hydroelectrica',
    'thermoelectric',
    'Energy Storage',
    'Battery',
    'Batteries',
    'Biogas',
    'Hydrogen',
    'renewable gas',
    'Landfill gas-to-energy',
    'Balance of System',
    'emissions'
    ]

dct = {}


df3 = df3.head(20)


#set different threshold for each sector/keyword

####start main program


for i in range (len(df3)):
    r = requests.get(df3.loc[i,"HTML_URL"], headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    tables = [
        [
            [td.get_text(strip=True) for td in tr.find_all('td')] 
            for tr in table.find_all('tr')
        ] 
        for table in soup.find_all('table')
    ]
    addy2 = "/" + tables[0][1][2]
    df3.loc[i,"HTML_URL"] = df3.loc[i,"HTML_URL"].replace('-', '')
    df3.loc[i,"HTML_URL"] = df3.loc[i,"HTML_URL"].replace('index.html', addy2)
    df3.loc[i,"HTML_URL"] = df3.loc[i,"HTML_URL"].replace('iXBRL', '')
    r = requests.get(df3.loc[i,"HTML_URL"], headers=HEADERS)
    soup2 = BeautifulSoup(r.content, 'html.parser')
    doc = soup2.text
    doc = doc.lower()
    start = doc.find('business')
    stop = doc.rfind('properties')
    section = doc[start:stop]

    keycount = 0
    dct[df3.loc[i,'ticker']] = []
    
    for f in keywords:
        wordnum = section.count(f.lower())
        dct[df3.loc[i,'ticker']].append(wordnum)
    
for i in range(0, len(dct)):
    total = 0
    for ele in range(0, len(dct[df3.loc[i,'ticker']])):
        total = total + dct[df3.loc[i,'ticker']][ele]
    wordcount.append(total)
 
#different threshhold for different sectors or keywords
  
for i in range(len(wordcount)):       
    if wordcount[i] > threshhold:
        match.append("clean")
    else:
        match.append("not_clean")
        

#final formatting

df6 = pd.DataFrame(dct)
df6_t = df6.transpose() 
df6_t.columns = keywords
df6_t["wordcount"] = wordcount
df6_t["is_clean"] = match
df7 = df6_t.loc[df6_t['is_clean'] == 'clean']

#create histogram and add vertical lines

'''
plt.hist(df1['wordcount'],bins=30)
plt.xlabel('Number of keyword matches')
plt.ylabel('Number of stocks')
plt.title('Histogram of keyword matches for liquid stocks')
plt.ylim(0,50)

df5 = pd.merge(df1, 
                      df4, 
                      on ='CIK', 
                      how ='inner')

for i in range (len(df5)):
    plt.axvline(df5.loc[i,"wordcount"], linestyle='dashed',linewidth=0.75, color="r")

 #only display clean names and save csv
'''


df7.to_csv('/Users/descalante/Anaconda3/clean_tickers_v2.csv')








