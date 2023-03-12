import sys
import EmbeddedLearn
import Clustering

# csvLoc = sys.argv[1]
csvLoc = 'output.csv'

all_vectors = EmbeddedLearn.embedWords(csvLoc, 'bert-base-uncased')
clusters = Clustering.dbscanClustering(all_vectors)
