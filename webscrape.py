import requests
from bs4 import BeautifulSoup
import csv

actor = "Jack Black"

# Define the URL of the Wikipedia page you want to scrape
url = "https://en.wikipedia.org/wiki/Kung_Fu_Panda_(film)"

# Make a request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract all paragraphs of text on the page
paragraphs = []
for paragraph in soup.find_all('p'):
    paragraphs.append(paragraph.text.strip())

# Write the extracted paragraphs of text to a CSV file
with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for paragraph in paragraphs:
        writer.writerow([actor, paragraph])