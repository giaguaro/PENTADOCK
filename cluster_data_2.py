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


pandarallel.initialize(progress_bar=True)

parser = argparse.ArgumentParser(description='Cluster molecules using k-means')
parser.add_argument('--input_file', type=str, help='input CSV file')
parser.add_argument('--n_clusters', type=int, default=100, help='number of clusters')
parser.add_argument('--output_prefix', type=str, help='output prefix')
parser.add_argument('--smiles_col', type=str, default='smiles', help='name of the SMILES column')
args = parser.parse_args()

device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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


# Load the data in batches of 10000 rows
batch_size = 2048
data_generator = pd.read_csv(args.input_file, sep=',', chunksize=batch_size)
clustered_train_data = pd.DataFrame()

# Cluster the molecules into groups using k-means clustering
n_clusters = args.n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

for i, (batch1, batch2) in enumerate(zip(data_generator, data_generator)):
    print(f"processing batches {i*2} and {i*2+1}")

    def generate_mol(smiles):
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception as e:
            print("Error occurred: ", e)
            return None

    # Convert the SMILES strings to RDKit molecules in parallel for both batches
    batch1['mol'] = batch1[args.smiles_col].parallel_apply(lambda x: generate_mol(x))
    batch2['mol'] = batch2[args.smiles_col].parallel_apply(lambda x: generate_mol(x))
    batch1.dropna(subset=['mol'], inplace=True)
    batch2.dropna(subset=['mol'], inplace=True)
    mols1 = list(batch1['mol'])
    mols2 = list(batch2['mol'])

    # Generate MAP4 fingerprints for each molecule in parallel for both batches
    fps1 = generate_fingerprints(mols1)
    fps2 = generate_fingerprints(mols2)

    # Compute pairwise similarity scores between all pairs of molecules using Faiss on both GPUs
    d = 1024
    xb1 = np.zeros((len(fps1), d), dtype='float32')
    xb2 = np.zeros((len(fps2), d), dtype='float32')
    for i, fp in enumerate(fps1):
        xb1[i] = fp.flatten()
    for i, fp in enumerate(fps2):
        xb2[i] = fp.flatten()
    res1 = faiss.StandardGpuResources()
    res2 = faiss.StandardGpuResources()
    gpu_index1 = faiss.GpuIndexFlatL2(res1, d)
    gpu_index2 = faiss.GpuIndexFlatL2(res2, d)
    cpu_index1 = faiss.IndexFlatL2(d)
    cpu_index2 = faiss.IndexFlatL2(d)

    # Add the fingerprints to the indexes on the two GPUs
    cpu_index1.add(xb1)
    cpu_index2.add(xb2)
    gpu_index1 = faiss.index_cpu_to_gpu(res1, 0, cpu_index1)
    gpu_index2 = faiss.index_cpu_to_gpu(res2, 1, cpu_index2)
    _, similarity_matrix1 = gpu_index1.search(xb1, len(xb1))
    _, similarity_matrix2 = gpu_index2.search(xb2, len(xb2))

    # Merge the similarity matrices from the two batches and cluster the molecules
    similarity_matrix = np.concatenate([similarity_matrix1, similarity_matrix2], axis=0)
    kmeans.fit(similarity_matrix)

    # Add a new column to each batch with the cluster labels
    batch1['cluster'] = kmeans.labels_[:len(batch1)]
    batch2['cluster'] = kmeans.labels_[len(batch1):]

    batch1 = batch1.drop('mol', axis=1)
    batch2 = batch2.drop('mol', axis=1)
    # Append the two batches to the clustered data
    clustered_train_data = pd.concat([clustered_train_data, batch1, batch2], ignore_index=True)


# Split each cluster into training, validation, testing, and predict sets, and store them in a dictionary
def split_data(cluster):
    num_mols = len(cluster)

    train_size = min(num_mols // 4, 500000)
    valid_size = min(num_mols // 8, 125000)
    test_size = min(num_mols // 8, 125000)

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


clustered_data = {i: split_data(clustered_train_data[clustered_train_data['cluster'] == i])
                  for i in range(n_clusters)}

train_data = pd.concat([clustered_data[i]['train'] for i in range(n_clusters)], ignore_index=True)
valid_data = pd.concat([clustered_data[i]['valid'] for i in range(n_clusters)], ignore_index=True)
test_data = pd.concat([clustered_data[i]['test'] for i in range(n_clusters)], ignore_index=True)
predict_data = pd.concat([clustered_data[i]['predict'] for i in range(n_clusters)], ignore_index=True)


# Shuffle the data
train_data = train_data.sample(frac=1, random_state=0).reset_index(drop=True)
valid_data = valid_data.sample(frac=1, random_state=0).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=0).reset_index(drop=True)
predict_data = predict_data.sample(frac=1, random_state=0).reset_index(drop=True)

# Save the data to CSV files
train_data.to_csv(args.output_prefix + '_train.smi', sep=' ', index=False)
valid_data.to_csv(args.output_prefix + '_valid.smi', sep=' ', index=False)
test_data.to_csv(args.output_prefix + '_test.smi', sep=' ', index=False)
predict_data.to_csv(args.output_prefix + '_predict.smi', sep=' ', index=False)

# Save the clustered data to a pickle file
#clustered_train_data.to_pickle(args.output_prefix + '_clustered.pkl')

print("Done!")
