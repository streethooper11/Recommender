import csv
import requests
import time
from bs4 import BeautifulSoup

#Put files of x y z actors that we have data for here, will aggregate it all into Movies.csv
files = ["Jack Black-movie.txt", "Morfydd Clark-movie.txt", "John Dimigallio-movie.txt"]

for file in files:
    #The text file will be links of X Y Z actor and named after them, for example Jack Black.txt
    actor = file.replace("-movie.txt","")
    time.sleep(1)
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

            print(page_text_lines)
            title = page_text_lines[12].strip()
            print(page_text_lines[12].strip())

            user_rating_index = page_text.find("User Rating") + len("User Rating")  # get the index of the start of the rating
            lines = page_text[user_rating_index:].splitlines()  # split the text into lines
            line_11 = lines[10].strip()  # select the 11th line after "User Rating" and remove leading/trailing white space

        # append the extracted line to the second column of the CSV file
        with open('Movies.csv', "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([actor, title , line_11])
