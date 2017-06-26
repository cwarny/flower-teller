from bs4 import BeautifulSoup
from urllib.parse import urlparse
import csv
import requests

with open('image-net.html') as infile, open('flowers.csv', 'w', newline='') as outfile:
	writer = csv.DictWriter(outfile, fieldnames=['wnid', 'name'])
	writer.writeheader()
	soup = BeautifulSoup(infile.read(), 'html.parser')
	lis = soup.find(id='3283').find('ul').find_all('li', recursive=False)
	for li in lis:
		writer.writerow({
			'wnid': urlparse(li.find('a')['href']).query.split('=')[-1],
			'name': li.find('a').get_text().strip().split(' (')[0]
		})
