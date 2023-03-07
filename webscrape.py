import requests
from bs4 import BeautifulSoup

# Define the URL of the Wikipedia page you want to scrape
url = "https://en.wikipedia.org/wiki/Albert_Einstein"

# Make a request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the first heading element with the text "Personality"
heading = soup.find('h2')

# Get the text of all the subsequent elements until the next heading element
text = ''
for sibling in heading.parent.find_next_siblings():
    if sibling.name == 'h2':
        break
    text += sibling.get_text()

# Print the extracted text
print(text)
