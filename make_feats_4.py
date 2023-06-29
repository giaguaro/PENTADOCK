import argparse
import dask.dataframe as dd
import dask.delayed
import pandas as pd
import pickle
import numpy as np
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from map4 import map4
from rdkit import Chem
from scipy import sparse
from tqdm.auto import tqdm
from pandarallel import pandarallel

generator = MakeGenerator(('rdkit2d',))
feats_columns = [name for name, numpy_type in generator.GetColumns()]

def generate_fingerprints(smiles, fp_type):
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == 'morgan':
        fp = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    elif fp_type == 'map4':
        fp = map4.MAP4Calculator(dimensions=1024, radius=2, is_folded=True).calculate(mol)
    else:
        raise ValueError('Invalid fingerprint type: {}'.format(fp_type))
    return fp.astype(np.float32)

@dask.delayed
def process_chunk(chunk, smiles_col, fp_type):
    pandarallel.initialize(progress_bar=True)
    applied_df = chunk[smiles_col].parallel_apply(generator.process).apply(pd.Series)
    fingerprints_df = chunk[smiles_col].parallel_apply(generate_fingerprints, fp_type=fp_type).apply(pd.Series)
    results = np.hstack((applied_df.iloc[:,1:].values.astype(np.float32), fingerprints_df.values.astype(np.float32)))
    return [sparse.csr_matrix(results[i]) for i in range(len(results))]

def process_file(input_file, output_file, smiles_col, fp_type, nb_workers):
    reader = dd.read_csv(input_file, sep=',', blocksize='64MB')
    chunks = reader.to_delayed()
    total_chunks = len(chunks)
    results = []

    delayed_results = [process_chunk(chunk, smiles_col, fp_type) for chunk in chunks]
    computed_results = dask.compute(*delayed_results, scheduler='processes', num_workers=nb_workers)

    for res in computed_results:
        results.extend(res)

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate rdkit2d and fingerprints and output the numpy array of those rdkit2d and fingerprints features (float32) as a pkl file without any other columns in the original input file.')
    parser.add_argument('input_file', type=str, help='path to the input csv file')
    parser.add_argument('output_file', type=str, help='path to the output pkl file')
    parser.add_argument('smiles_col', type=str, help='column name of the SMILES in the input csv file')
    parser.add_argument('fp_type', type=str, choices=['morgan', 'map4'], help='type of fingerprint to generate')
    parser.add_argument('nb_workers', type=int, help='number of workers to parallize the operation across')
    args = parser.parse_args()
    pandarallel.initialize(progress_bar=True)
    process_file(args.input_file, args.output_file, args.smiles_col, args.fp_type, args.nb_workers)

