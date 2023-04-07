import requests
from bs4 import BeautifulSoup
import csv


#Put files of x y z actors that we have data for here, will aggregate it all into Movies.csv
files = ["Jack Black-actor.txt","Morfydd Clark-actor.txt", "Jay Baruchel-actor.txt", "Seth MacFarlane-actor.txt", "Lawrence-actor.txt", "Keanu Reeves-actor.txt", "John Dimigallio-actor.txt"]

for file in files:
    #The text file will be links of X Y Z actor and named after them, for example Jack Black.txt
    actor = file.replace("-actor.txt","")

    #Find the URL of the Wikipedia page you want to scrape
    with open(file, "r") as f:
        urls = f.readlines()

        for url in urls:
            
            url = url.strip()  # remove leading/trailing white space

            # Make a request to the URL
            response = requests.get(url)

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            title = ["overview", "personality", "character", "characterization", "biography"]
            paragraphs = []

            first_paragraph = soup.find('p')
            if first_paragraph:
                paragraphs.append(first_paragraph.text.strip())

            # Extract all paragraphs of text on the page
            for header in soup.find_all('h2'):
                header_text = header.text.strip().replace("[edit]", "")
                if header_text.lower() in title:
                    print(header_text)
                    next_paragraph = header.find_next('p')  # find the next paragraph after the header
                    if next_paragraph:
                        paragraphs.append(next_paragraph.text.strip())

            # Write the extracted paragraphs of text to a CSV file
            with open('Roles.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for paragraph in paragraphs:
                    writer.writerow([actor, paragraph])