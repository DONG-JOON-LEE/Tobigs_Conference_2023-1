import os
import re
import requests

from inspect import getfile
from urllib import request
from urllib.error import HTTPError
from bs4 import BeautifulSoup
from urllib.request import urlopen
from tqdm import tqdm

def main():
    temp = [] 
    url = [] 

    html = urlopen("https://freesound.org/search/?advanced=0&g=1&only_p=&q=subway&f=channels%3A%222%22+type%3A%22wav%22&s=Automatic+by+relevance&w=&page=4#sound")  
    bsObject = BeautifulSoup(html, "html.parser")

    for link in bsObject.find_all('a'):
        temp.append(str(link.get('href')))

        
    p = re.compile('.*[.]mp3')
    for href in temp:
        m = p.match(href)
        if m:
            url.append(str(m.group()))

    for i, url_ture in enumerate(url):
        request.urlretrieve(url_ture, '{}.mp3'.format(i))

if __name__ == "__main__":
    main()