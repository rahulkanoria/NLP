# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import urllib.request
from bs4 import BeautifulSoup

url = "http://www.rajbhasha.nic.in/"
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html, features="lxml")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

# get text
text = soup.get_text()
text.encode('utf-8')
text_file = open("Output.txt", "w")
text_file.write(text)
text_file.close
print(text)