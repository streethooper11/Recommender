import requests
from bs4 import BeautifulSoup
import csv
import sys
sys.stdout.encoding = 'utf-8'

# Define the URL of the Wikipedia page you want to scrape
url = "https://en.wikipedia.org/wiki/Kung_Fu_Panda_(film)"

# Make a request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the first heading element with the text "Personality"
heading = soup.find('h2')

# Get the text of all the subsequent elements until the next heading element

# Extract all paragraphs of text on the page
paragraphs = []
for paragraph in soup.find_all('p'):
    paragraphs.append(paragraph.text.strip())

# Write the extracted paragraphs of text to a CSV file
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    for paragraph in paragraphs:
        writer.writerow([paragraph])