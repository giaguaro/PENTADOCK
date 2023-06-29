import argparse
import pickle

parser = argparse.ArgumentParser(description='Combine two pickle files into one')
parser.add_argument('file1', type=str, help='path to the first input pickle file')
parser.add_argument('file2', type=str, help='path to the second input pickle file')
parser.add_argument('output_file', type=str, help='path to the output pickle file')
args = parser.parse_args()

# Load the first pickle file
with open(args.file1, 'rb') as f:
    data1 = pickle.load(f)

# Load the second pickle file
with open(args.file2, 'rb') as f:
    data2 = pickle.load(f)

# Concatenate the results
data_combined = data1 + data2

# Write the combined results to a new pickle file
with open(args.output_file, 'wb') as f:
    pickle.dump(data_combined, f)

