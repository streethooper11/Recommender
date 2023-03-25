import csv
import requests
from bs4 import BeautifulSoup

#Put files of x y z actors that we have data for here, will aggregate it all into Movies.csv
files = ["Jack Black-movie.txt",]

for file in files:
    #The text file will be links of X Y Z actor and named after them, for example Jack Black.txt
    actor = file.replace("-movie.txt","")

    with open(file, "r") as f:
        urls = f.readlines()

        for url in urls:
            url = url.strip()  # remove leading/trailing white space

            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            page_text = soup.text
            page_text_lines = page_text.splitlines()

            title = page_text_lines[93].strip()

            user_rating_index = page_text.find("User Rating") + len("User Rating")  # get the index of the start of the rating
            lines = page_text[user_rating_index:].splitlines()  # split the text into lines
            line_11 = lines[10].strip()  # select the 11th line after "User Rating" and remove leading/trailing white space

            # append the extracted line to the second column of the CSV file
            with open('Movies.csv', "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([actor, title , line_11])
