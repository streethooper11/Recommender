#!/usr/bin/env python3
"""
This is the executable file that webscrapes actor and movie information for training data
Source for listing file in a folder:
https://www.geeksforgeeks.org/python-list-files-in-a-directory/
"""
import os
from Logic import webscrape

def webscrapeTrainData():
    actorsPath = "Data/TrainData/Actors/"  # All training actor files are stored here
    moviesPath = "Data/TrainData/Movies/"  # All training movie files are stored here

    filesActorsTemp = os.listdir(actorsPath)
    filesActors = [actorsPath + actorName for actorName in filesActorsTemp]

    filesMoviesTemp = os.listdir(moviesPath)
    filesMovies = [moviesPath + movieName for movieName in filesMoviesTemp]

    webscrape.webscrapeActors(filesActors)
    #    webscrape.webscrapeMovies(filesMovies)

if __name__ == "__main__":
    webscrapeTrainData()
