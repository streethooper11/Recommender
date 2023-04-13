## Associative Actrec
Actor recommendation system using BERT, a clustering method, and a custom scoring system.

This program is developed in Python 3.11.

### How to use
#### Requirements
The following python packages are required to run the program:
1. numpy
2. pandas
3. torch
4. transformers

#### The program is divided into 3 parts:
1. Run part1_webscrapeTrainData.py to extract role descriptions and and movie ratings from the web
2. Run part2_trainDataEmbed.py to use extracted data from part1 to create word embeddings for role descriptions from\
the training data.\
Run part2_manualTrainDataEmbed.py to use manually gathered role descriptions (TrainData\RolesManual.csv) to create word\
embeddings instead. The manual one currently includes more information.
3. Run part3_inputDataEmbedToRank.py to create word embeddings for role descriptions from the test data
and combine the embeddings with the training data to generate actor recommendations.\
Run part3_manualInputDataEmbedToRank.py to use manually gathered movie ratings (TrainData\MoviesManual.csv) to create movie\
rating data instead. The manual one currently includes more information.

#### You are able to execute each part separately, and only some of the parts in order to increase efficiency:
1. After running part 1, you can run part 2 multiple times to re-use the scraped csv files.
2. After running part 2, you can run part 3 multiple times to re-use the word embeddings from the training data.

#### Notes
Running part 3 shows, for each testing role description, the recommended actors and whether or not the actual actor
that participated in that role is found in the recommendation on the console screen.\
When all the testing data is done processing, it displays the accuracy for all the data given.

part3test_PlotClusters.py was used after creating part 2 to draw a bar graph after clustering:\
One graph for cluster results in all data and one graph for cluster results in testing data only.
