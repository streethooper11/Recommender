#!/usr/bin/env python3
"""
This is the executable file that webscrapes actor and movie information for testing data
Source for listing file in a folder:
https://www.geeksforgeeks.org/python-list-files-in-a-directory/
"""
import os
from Logic import webscrape

def webscrapeInputData():
    actorsPath = "Data/TestData/Actors/"  # All input actor files are stored here

    filesActorsTemp = os.listdir(actorsPath)
    filesActors = [actorsPath + actorName for actorName in filesActorsTemp]

    outputPath = "Data/TestData/InputDescription.csv"
    webscrape.webscrapeActors(filesActors, outputPath)

if __name__ == "__main__":
    webscrapeInputData()
