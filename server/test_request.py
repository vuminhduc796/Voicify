import requests
import urllib.parse

url = 'http://localhost:8099/parse/user_study'
while True:
    utterance = input()
    print(requests.get(url, params={'q': utterance}).json())
