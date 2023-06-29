import argparse
import pickle
import pandas as pd
import numpy as np
import faiss
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
from map4 import map4
import multiprocessing
import torch

from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Cluster molecules using k-means')
parser.add_argument('--input_file', type=str, help='input SMI file')
parser.add_argument('--n_clusters', type=int, default=250, help='number of clusters')
parser.add_argument('--output_prefix', type=str, help='output prefix')
parser.add_argument('--smiles_col', type=str, default='smiles', help='name of the SMILES column')
parser.add_argument('--workers', type=int, default=24, help='number of workers')
parser.add_argument('--frequency', type=int, default=10, help='Cluster every ..')
args = parser.parse_args()

pandarallel.initialize(progress_bar=False)

#device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def plot_centroids_history(centroids_history, n_clusters):
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        centroid_coords = np.array([centroids[i] for centroids in centroids_history])
        ax.plot(centroid_coords[:, 0], centroid_coords[:, 1], marker='o', linestyle='-', label=f'Cluster {i + 1}')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Cluster Centroids History')
    ax.legend()
    plt.show()
    # Save the plot
    plt.savefig(f'kmeans_cluster_history.png', dpi=300)
    plt.clf()

# Define a function to generate Morgan fingerprints for a batch of molecules
def generate_fingerprints(mols):
    fps = []
    pool = multiprocessing.Pool(processes=45)
    fps = pool.map(map4.MAP4Calculator(dimensions=1024, radius=2, is_folded=True).calculate, mols)
    pool.close()
    pool.join()
    fps_dense = [fp.astype(np.float32) for fp in fps]
    fps_np = np.stack(fps_dense)
    return fps_np

# Split each cluster into training, validation, testing, and predict sets, and store them in a dictionary
def split_data(cluster):
    num_mols = len(cluster)

    train_size = min(num_mols // 4, 50000)
    valid_size = min(num_mols // 16, 12500)
    test_size = min(num_mols // 16, 12500)

    stratify_by = np.array(cluster['cluster'])
    train, test = train_test_split(cluster, test_size=valid_size + test_size, random_state=0, stratify=stratify_by)
    stratify_by = np.array(test['cluster'])
    valid, test = train_test_split(test, test_size=test_size, random_state=0, stratify=stratify_by)

    predict_size = num_mols - train_size - valid_size - test_size

    return {
        'train': train.head(train_size) if len(train) > train_size else train,
        'valid': valid.head(valid_size) if len(valid) > valid_size else valid,
        'test': test.head(test_size) if len(test) > test_size else test,
        'predict': cluster.tail(predict_size) if len(cluster) > predict_size else cluster
    }



#import random

#input_file = args.input_file
#shuffled_input_file = 'shuffled_' + input_file

#with open(input_file, 'r') as source:
#    lines = source.readlines()

#header, lines = lines[0], lines[1:]
#random.shuffle(lines)

#with open(shuffled_input_file, 'w') as destination:
#    destination.write(header)
#    destination.writelines(lines)


# Load the data in batches of 10000 rows
batch_size = 50000
data_generator = pd.read_csv(args.input_file, sep=' ', chunksize=batch_size)
clustered_train_data = pd.DataFrame()

# Cluster the molecules into groups using k-means clustering
n_clusters = args.n_clusters
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init='k-means++', n_init='auto', max_iter=100, batch_size=batch_size, reassignment_ratio=0.001)

# Initialize empty dataframes
train_data = pd.DataFrame()
valid_data = pd.DataFrame()
test_data = pd.DataFrame()
predict_data = pd.DataFrame()

temp_df = pd.DataFrame()

# Add counters for train, valid, and test samples
train_count, valid_count, test_count = 0, 0, 0
max_train, max_valid, max_test = 50000, 12500, 12500

# Initialize a list to store the batch data
batch_data_list = []
similarity_matrix_all = None

centroids_history = []
dump_frequency = 20
count=0
# Change the loop condition to stop when predefined numbers are reached
for i, batch in enumerate(data_generator):
    if train_count >= max_train and valid_count >= max_valid and test_count >= max_test:
        break
    print("train data size: ", len(train_data), ", valid data size: ", len(valid_data),", test data size: ", len(test_data))
    print(f"processing batches {i}")

    def generate_mol(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception as e:
            print("Error occurred: ", e)
            return None

    # Convert the SMILES strings to RDKit molecules in parallel for both batches
    batch['mol'] = batch[args.smiles_col].parallel_apply(lambda x: generate_mol(x))
    batch.dropna(subset=['mol'], inplace=True)
    mols = list(batch['mol'])

    # Generate MAP4 fingerprints for each molecule in parallel for both batches
    fps = generate_fingerprints(mols)
    kmeans.partial_fit(fps)

    centroids_history.append(kmeans.cluster_centers_.copy())

    count+=1
    if (count + 1) % dump_frequency == 0:

        batch['cluster'] = kmeans.predict(fps)
        #batch_data_list.append(batch)
        # Concatenate the batch data from the list
        #combined_batches = pd.concat(batch_data_list, ignore_index=True)
        combined_batches = batch
        combined_batches = combined_batches.drop('mol', axis=1)

        # Check if the least populated class has at least 2 members
        stratify_by = np.array(combined_batches['cluster'])
        #stratify_by = np.array(temp_df['cluster'])
        print("min bincount for least represented cluster: ", np.min(np.bincount(stratify_by)))
        if np.min(np.bincount(stratify_by)) >= args.frequency and np.min(np.bincount(stratify_by)) % 2 == 0:
            # Split and update the DataFrames if the condition is met
            cluster_data = split_data(combined_batches)
            train_data = pd.concat([train_data, cluster_data['train']], ignore_index=True)
            valid_data = pd.concat([valid_data, cluster_data['valid']], ignore_index=True)
            test_data = pd.concat([test_data, cluster_data['test']], ignore_index=True)
            predict_data = pd.concat([predict_data, cluster_data['predict']], ignore_index=True)

            # Reset the temporary DataFrames
            combined_batches = pd.DataFrame()
            batch_data_list = []
            # Update the counters
            train_count += len(cluster_data['train'])
            valid_count += len(cluster_data['valid'])
            test_count += len(cluster_data['test'])

plot_centroids_history(centroids_history, n_clusters)
# Shuffle the data
train_data = train_data.sample(frac=1, random_state=0).reset_index(drop=True)
valid_data = valid_data.sample(frac=1, random_state=0).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=0).reset_index(drop=True)

# Save the data to SMI files
train_data.iloc[:, :2].to_csv(args.output_prefix + '_train.smi', sep=' ', index=False)
valid_data.iloc[:, :2].to_csv(args.output_prefix + '_valid.smi', sep=' ', index=False)
test_data.iloc[:, :2].to_csv(args.output_prefix + '_test.smi', sep=' ', index=False)

# Save the data to CSV files
train_data.to_csv(args.output_prefix + '_train.csv', sep=',', index=False)
valid_data.to_csv(args.output_prefix + '_valid.csv', sep=',', index=False)
test_data.to_csv(args.output_prefix + '_test.csv', sep=',', index=False)
# Add the remaining unprocessed data to the predict dataset
##for batch in data_generator:
##    predict_data = pd.concat([predict_data, batch], ignore_index=True)

#clustered_data = {i: split_data(clustered_train_data[clustered_train_data['cluster'] == i])
#                  for i in range(n_clusters)}

#train_data = pd.concat([clustered_data[i]['train'] for i in range(n_clusters)], ignore_index=True)
#valid_data = pd.concat([clustered_data[i]['valid'] for i in range(n_clusters)], ignore_index=True)
#test_data = pd.concat([clustered_data[i]['test'] for i in range(n_clusters)], ignore_index=True)
#predict_data = pd.concat([clustered_data[i]['predict'] for i in range(n_clusters)], ignore_index=True)


# Shuffle the data
#predict_data = predict_data.sample(frac=1, random_state=0).reset_index(drop=True)

# Save the data to CSV files
#predict_data.to_csv(args.output_prefix + '_predict.smi', sep=' ', index=False)

# Save the clustered data to a pickle file
#clustered_train_data.to_pickle(args.output_prefix + '_clustered.pkl')

print("Done!")
