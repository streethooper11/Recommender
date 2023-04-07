import csv
import requests
from bs4 import BeautifulSoup

url = "https://www.imdb.com/title/tt0111161/ratings/?ref_=tt_ov_rt"  # replace with the IMDb URL of the movie you want to scrape
output_file = "output2.csv"  # replace with the name of the output CSV file
actor = "Jack Black"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

page_text = soup.text
user_rating_index = page_text.find("User Rating") + len("User Rating")  # get the index of the start of the rating
lines = page_text[user_rating_index:].splitlines()  # split the text into lines
print(lines)
line_11 = lines[10].strip()  # select the 11th line after "User Rating" and remove leading/trailing white space

# append the extracted line to the second column of the CSV file
with open(output_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([actor, line_11])