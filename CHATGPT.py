# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:08:18 2023

@author: descalante
"""

#DANIEL TALK TO CHATGPT

import openai
#from sec_api import ExtractorApi

openai.api_key = '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'



#extractorApi = ExtractorApi("141548b6702631e82e2c0d32f392c4ca8e9d33aa5dfe83e2ebc0b096b7f6829a")


#filing_url = "https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm"

#section_text = extractorApi.get_section(filing_url, "1A", "text")

messages = [
    {"role": "system", "content": "you are a helpful assistant"},
]
     

#message = input(f"User : Summarize the following text in 15 sentencens:\n{section_text} ")

while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})