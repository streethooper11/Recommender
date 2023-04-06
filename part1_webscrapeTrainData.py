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
    outputPathActors = "Data/TrainData/Roles.csv"

    filesMoviesTemp = os.listdir(moviesPath)
    filesMovies = [moviesPath + movieName for movieName in filesMoviesTemp]
    outputPathMovies = "Data/TrainData/Movies.csv"

    webscrape.webscrapeActors(filesActors, outputPathActors)
    webscrape.webscrapeMovies(filesMovies, outputPathMovies)

if __name__ == "__main__":
    webscrapeTrainData()
