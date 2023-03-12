import sys
import EmbeddedLearn
import Clustering

# csvLoc = sys.argv[1]
csvLoc = 'output.csv'

all_vectors = EmbeddedLearn.embedWords(csvLoc, 'bert-base-uncased')
data_with_clusters = Clustering.dbscanClustering(all_vectors)
