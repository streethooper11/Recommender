import time
import requests
from bs4 import BeautifulSoup
import csv


def webscrapeActors(files: list, output: str):
    for file in files:
        # The text file will be links of X Y Z actor and named after them, for example Jack Black.txt
        actor = file[22:-10]  # remove Data/TrainData/Actors/ and -movie.txt part

        with open(output, mode="a", newline="", encoding="utf-8") as writeFile:
            # Find the URL of the Wikipedia page you want to scrape
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
                            next_paragraph = header.find_next('p')  # find the next paragraph after the header
                            if next_paragraph:
                                paragraphs.append(next_paragraph.text.strip())

                    # Write the extracted paragraphs of text to a CSV file
                    writer = csv.writer(writeFile)
                    for paragraph in paragraphs:
                        writer.writerow([actor, paragraph])


def webscrapeMovies(files: list, output: str):
    for file in files:
        # The text file will be links of X Y Z actor and named after them, for example Jack Black.txt
        actor = file[22:-10]  # remove Data/TrainData/Movies/ and -movie.txt part

        with open(output, "a", newline="") as writeFile:
            with open(file, "r") as f:
                urls = f.readlines()

                for url in urls:
                    time.sleep(1)  # add a delay of 1 second

                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
                    }
                    response = requests.get(url, headers=headers)
                    soup = BeautifulSoup(response.text, "html.parser")

                    page_text = soup.text
                    page_text_lines = page_text.splitlines()

                    title = page_text_lines[12].strip()

                    user_rating_index = page_text.find("User Rating") + len(
                        "User Rating")  # get the index of the start of the rating
                    lines = page_text[user_rating_index:].splitlines()  # split the text into lines
                    line_11 = lines[
                        10].strip()  # select the 11th line after "User Rating" and remove leading/trailing white space\

                    # write the extracted line to the second column of the CSV file
                    writer = csv.writer(writeFile)
                    writer.writerow([actor, title, line_11])
