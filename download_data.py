import requests
import wget

url = 'http://mattmahoney.net/dc/text8.zip'
fname = 'text8.zip'

'''
r = requests.get(url, allow_redirects=True)

open(fname, 'wb').write(r.content)
'''

wget.download(url, fname)