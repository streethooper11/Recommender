import EmbeddedLearn
import Clustering
import ProcessList

# csvLoc = sys.argv[1]
csvLoc = 'output.csv'

all_actors, all_subwords, all_vectors = EmbeddedLearn.embedWords(csvLoc, 'bert-base-uncased')
cluster_data = Clustering.dbscanClustering(all_vectors)

cluster_actor_dict = ProcessList.createDictionary_ClustersAndActors(cluster_data, all_actors)
