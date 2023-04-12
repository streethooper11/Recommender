## Associative Actrec
Actor recommendation system using BERT, a clustering method, and a custom scoring system.

### How to use
#### The program is divided into 3 parts:
1. Run part1_webscrapeTrainData.py to read files inside TrainData/Actors folder to extract role descriptions and\
files inside TrainData\Movies folder to extract movie ratings
2. Run part2_trainDataEmbed.py to use extracted data from part1 to create word embeddings for role descriptions from\
the training data.\
Run part2_manualTrainDataEmbed.py to use manually gathered role descriptions (TrainData\RolesManual.csv) to create word\
embeddings. The manual one currently includes more information.
3. Run part3_inputDataEmbedToRank.py to read files inside TestData/InputDescriptionManual.csv to create word embeddings\
for role descriptions from the test data, and combines the embedding with the training data to generate\
actor recommendations, using movie ratings data from part 1 to calculate ratings score.\
Or run part3_manualInputDataEmbedToRank.py to use manually gathered movie ratings (TrainData\MoviesManual.csv) to create movie\
rating data. The manual one currently includes more information.

You may choose the clustering method between DBSCAN and K-Means by commenting/uncommenting\
the indicated lines at the bottom of the part 3 file. It is defaulted at DBSCAN.\
You may also change hyperparameters by changing the values at the bottom of the part 3 file.

#### You are able to execute each part separately, and only some of the parts in order to increase efficiency:
1. After running part 1, you can run part 2 multiple times to re-use the scraped csv files.
2. After running part 2, you can run part 3 multiple times to re-use the word embeddings from the training data.

#### Output
Running part 3 shows, for each testing role description, the recommended actors and whether or not the actual actor
that participated in that role is found in the recommendation on the console screen.\
When all the testing data is done processing, it displays the accuracy for all the data given.

part3test_PlotClusters.py can be used after creating part 2; it takes a random sample testing data,
clusters data in DBSCAN and K-Means clustering,
and plots the cluster results with the number of words in each cluster:\
Two graphs for cluster results in training and testing data for each clustering method,\
Two graphs for cluster results in testing data only for each clustering method.
