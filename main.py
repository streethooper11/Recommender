import sys
import EmbeddedLearn
import Clustering

# csvLoc = sys.argv[1]
csvLoc = 'output.csv'

all_actors, all_subwords, all_vectors = EmbeddedLearn.embedWords(csvLoc, 'bert-base-uncased')
cluster_data = Clustering.dbscanClustering(all_vectors)
