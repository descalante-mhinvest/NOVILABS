# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:22:35 2023

@author: descalante
"""


#imports
import requests
import edgar
import pandas as pd
from bs4 import BeautifulSoup
'''import matplotlib.pyplot as plt'''
import re
import time



HEADERS = {'User-Agent': "Miller Howard Investments descalante@mhinvest.com"}
#tickers_cik = requests.get("https://www.sec.gov/Archives/edgar/data/1084869/0001437749-22-025012-index.html", headers=HEADERS)
#tickers_cik = pd.json_normalize(pd.json_normalize(tickers_cik.json(),\max_level=0).values[0])

#download and read the edgar index, data cleaning
edgar.download_index('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3', 2021,'descalante@mhinvest.com',skip_all_present_except_last=False)

class QuarterData:
    def __init__(self, quarter_num, year):
        self.quarter_num = quarter_num
        self.year = year
        self.df = None
        
    def clean(self):
        df = pd.read_csv(f"U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/{self.year}-QTR{self.quarter_num}.tsv", sep="|")
        df.columns =['CIK', 'Company_Name', 'Form_Type', 'Filing_Date', 'TXT_URL', 'HTML_URL']
        df = df[df.Form_Type.isin(["DEFC14A","DEF 14A"])]
        self.df = df
        return self.df

df0 = QuarterData(3,2023).clean()
df = QuarterData(2,2023).clean()
df2 = QuarterData(1,2023).clean()
df3 = QuarterData(4,2022).clean()
df4 = QuarterData(3,2022).clean()
df5 = QuarterData(2,2022).clean()

# Concatenate the dataframes
df_all = pd.concat([df0, df, df2, df3, df4, df5])

def clean_dataframe(df, rename_cik=False, fix_cik=False, int_cik=False, in_addy=False):
    if 'Filing_Date' in df.columns:
        df['Filing_Date'] = pd.to_datetime(df['Filing_Date'])
        df = df.sort_values('Filing_Date').drop_duplicates('Company_Name', keep='last')
    if fix_cik:
        if 'CIK' in df.columns:
            df['CIK'] = df['CIK'].astype(str).apply(lambda x: x.zfill(10))
        if 'CIK_NUMBER' in df.columns:
            df['CIK_NUMBER'] = df['CIK_NUMBER'].astype(str).apply(lambda x: x.zfill(10))
    if rename_cik:
        df = df.rename(columns={'CIK_NUMBER': 'CIK'})
    if int_cik:
        df['CIK'] = df['CIK'].astype(int)
    if in_addy:
        addy = 'https://www.sec.gov/Archives/'
        df['HTML_URL'] = addy + df['HTML_URL']
        df['HTML_URL'] = df['HTML_URL'].astype(str)
        df['HTML_URL'] = df['HTML_URL'].str.strip()
    return df

#cik number dataframe
df6 = pd.read_csv("U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/R2K_Q2END.csv")

#r2k dataframe
df7 = pd.read_csv("U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/CIK_NUMBERS_072123.csv")

#EDIT CIK DATA
df7["TICKER"] = df7["TICKER"].str.replace(' Equity', '')
df8 = pd.merge(df6, df7, on ='TICKER', how ='inner')
df_all = clean_dataframe(df_all, fix_cik=True, in_addy=True)
df8 = clean_dataframe(df8, fix_cik=True, rename_cik=True)
df_r2k = pd.merge(df8, df_all, on ='CIK', how ='inner')
df_r2K = df_r2k.drop(columns=['NAME_x', 'NAME_y','TXT_URL'])

# Find 'TICKER' values in df6 that are not in df7
missing_in_df7 = df6.loc[~df6['TICKER'].isin(df7['TICKER'])]
# Find 'CIK' values in df8 that are not in df_all
missing_in_df_all = df8.loc[~df8['CIK'].isin(df_all['CIK'])]
print(repr(df_r2k.loc[5, "HTML_URL"]))
#test = requests.get(df_r2k.loc[5, "HTML_URL"], headers=HEADERS)
from retrying import retry

test_results = []
#proxy = "http://61.233.25.166:80"
#proxies = {"http": proxy,"https": proxy}
for i in range(len(df_r2k)):
    test = requests.get(df_r2k.loc[i, "HTML_URL"], headers=HEADERS)  #proxies=proxies
    time.sleep(15)
    if test.status_code == 200:
           test_results.append((i, "Success"))
    else:
        test_results.append((i, "Failed with status code: " + str(test.status_code)))

'''

url = 'https://www.sec.gov/Archives/edgar/data/1084869/0001437749-22-025012-index.html'
HEADERS = {'User-Agent': "descalante@mhinvest.com"}
#HEADERS = {'User-Agent': 'Miller Howard descalante@mhinvest.com', 'Accept-Encoding': 'gzip,deflate', 'Host': 'www.sec.gov'}
r = requests.get(url, HEADERS)
soup = BeautifulSoup(r.content, 'html.parser')
tables = [
    [
        [td.get_text(strip=True) for td in tr.find_all('td')] 
        for tr in table.find_all('tr')
    ] 
    for table in soup.find_all('table')
]
addy2 = "/" + tables[0][1][2]
'''
'''
class DocumentAnalyzer:
    def __init__(self, data_frame, threshhold=10):
        self.df = data_frame
        self.threshhold = threshhold
        self.keywords = [
            'founder of x',
            'co-founder of x',
            'founder of the company',
            'co-founded x',
            'co-founder of the company',
            'our co-founder',
            'co-founder of x'
        ]

    def analyze_documents(self):
        wordcount = []
        match = []
        dct = {}
                
        

        #@retry(stop_max_attempt_number=5, wait_fixed=20000)  # Retry up to 5 times, wait 20 seconds between attempts
        #def get_page(url):
            #response = requests.get(url, headers=HEADERS)
            #return response.content
        
        # Then in your loop
        for i in range(len(self.df)):
            #url = self.df.loc[i,"HTML_URL"]
            #content = get_page(url)
            print(self.df.loc[i,'HTML_URL'])
            content = requests.get(self.df.loc[i,"HTML_URL"], headers=HEADERS)
            #content = requests.get(self.df.loc[1,"HTML_URL"], headers=HEADERS)
            time.sleep(15)
            soup = BeautifulSoup(content, 'html.parser')
            tables = [
                [
                    [td.get_text(strip=True) for td in tr.find_all('td')] 
                    for tr in table.find_all('tr')
                ] 
                for table in soup.find_all('table')
            ]
            addy2 = "/" + tables[0][1][2]
            self.df.loc[i,"HTML_URL"] = self.df.loc[i,"HTML_URL"].replace('-', '')
            self.df.loc[i,"HTML_URL"] = self.df.loc[i,"HTML_URL"].replace('index.html', addy2)
            self.df.loc[i,"HTML_URL"] = self.df.loc[i,"HTML_URL"].replace('iXBRL', '')
            content2 = requests.get(self.df.loc[i,"HTML_URL"], headers=HEADERS)
            #content2 = get_page(self.df.loc[i,"HTML_URL"])
            time.sleep(15)
            soup2 = BeautifulSoup(content2, 'html.parser')
            doc = soup2.text
            doc = doc.lower()
            start = doc.find('director nominees')
            stop = doc.rfind('related party transaction')
            section = doc[start:stop]
            dct[self.df.loc[i,'TICKER']] = []
            
            for f in self.keywords:
                # Replace 'x' with company name in keyword
                full_company_name = self.df.loc[i, 'Company_Name']
                company_name = re.sub(r"( Inc.| corp| /DE/).*", "", full_company_name, flags=re.I)
                keyword = f.replace('x', company_name)
                wordnum = section.count(keyword.lower())
                dct[self.df.loc[i,'TICKER']].append(wordnum)

        for ticker in dct:
            total = sum(dct[ticker])
            wordcount.append(total)
 
        for i in range(len(wordcount)):       
            if wordcount[i] > self.threshhold:
                match.append("Founder_led")
            else:
                match.append("Not_Founder_led")
                
        #final formatting
        df10 = pd.DataFrame(dct)
        df10_t = df10.transpose()
        

        df10_t.columns = self.keywords
        df10_t["wordcount"] = wordcount
        df10_t["IS_FOUNDED"] = match
        df11 = df10_t.loc[df10_t['IS_FOUNDED'] == 'Founder_led']

        return df11

df_r2k = df_r2k.head(20)
analyzer = DocumentAnalyzer(df_r2k)
result = analyzer.analyze_documents()
'''
'''
#set threshhold, initalize lists, dict
wordcount = []
match = []
threshhold = 2
dct = {}
df_r2k = df_r2k.head(20)

#Keywords list, not case sensitive
keywords = ['founder','founded','founder\'s']
HEADERS = {'User-Agent': 'Miller Howard Investments descalante@mhinvest.com'}
for i in range (len(df_r2k)):
    r = requests.get(df_r2k.loc[i,"HTML_URL"], headers=HEADERS)
    soup = BeautifulSoup(r.content, 'html.parser')
    tables = [
        [
            [td.get_text(strip=True) for td in tr.find_all('td')] 
            for tr in table.find_all('tr')
        ] 
        for table in soup.find_all('table')
    ]
    addy2 = "/" + tables[0][1][2]
    df_r2k.loc[i,"HTML_URL"] = df_r2k.loc[i,"HTML_URL"].replace('-', '')
    df_r2k.loc[i,"HTML_URL"] = df_r2k.loc[i,"HTML_URL"].replace('index.html', addy2)
    df_r2k.loc[i,"HTML_URL"] = df_r2k.loc[i,"HTML_URL"].replace('iXBRL', '')
    r = requests.get(df_r2k.loc[i,"HTML_URL"], headers=HEADERS)
    soup2 = BeautifulSoup(r.content, 'html.parser')
    doc = soup2.text
    doc = doc.lower()
    start = doc.find('director nominees')
    stop = doc.rfind('related party transaction')
    section = doc[start:stop]

    keycount = 0
    dct[df_r2k.loc[i,'TICKER']] = []
    
    for f in keywords:
        wordnum = section.count(f.lower())
        dct[df_r2k.loc[i,'TICKER']].append(wordnum)
    
for ticker in dct:
    total = sum(dct[ticker])
    wordcount.append(total)
 
#different threshhold for different sectors or keywords
for i in range(len(wordcount)):       
    if wordcount[i] > threshhold:
        match.append("Founder_led")
    else:
        match.append("Not_Founder_led")
        
#final formatting
df10 = pd.DataFrame(dct)
df10_t = df10.transpose() 
df10_t.columns = keywords
df10_t["wordcount"] = wordcount
df10_t["IS_FOUNDED"] = match
df11 = df10_t.loc[df10_t['IS_FOUNDED'] == 'Founder_led']
'''

'''
df.to_csv('U:/03_InvestmentTeamMembers/12_DAE/.spyder-py3/founded_or_not.csv')
'''







