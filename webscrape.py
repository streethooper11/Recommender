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

title = ["overview", "personality", "character"]
paragraphs = []

# Extract all paragraphs of text on the page
for header in soup.find_all('h2'):
    header_text = header.text.strip().replace("[edit]", "")
    if header_text.lower() in title:
        print(header_text)
        next_paragraph = header.find_next('p')  # find the next paragraph after the header
        if next_paragraph:
            paragraphs.append(next_paragraph.text.strip())

# Write the extracted paragraphs of text to a CSV file
with open('output.csv', mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for paragraph in paragraphs:
        writer.writerow([actor, paragraph])