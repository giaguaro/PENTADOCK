import argparse
import pandas as pd

def min_max_scaler(data, column, reverse=False):
    min_value = data[column].min()
    max_value = data[column].max()

    if reverse:
        data[column] = data[column] * (max_value - min_value) + min_value
    else:
        data[column] = (data[column] - min_value) / (max_value - min_value)

    return data

def main(args):
    data = pd.read_csv(args.input_file)
    
    if args.operation == 'normalize':
        scaled_data = min_max_scaler(data, args.score_column, reverse=False)
    elif args.operation == 'denormalize':
        scaled_data = min_max_scaler(data, args.score_column, reverse=True)
    else:
        raise ValueError("Invalid operation. Choose 'normalize' or 'denormalize'.")

    scaled_data.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("score_column", help="Name of the score column to normalize or denormalize")
    parser.add_argument("output_file", help="Path to the output CSV file")
    parser.add_argument("operation", choices=['normalize', 'denormalize'], help="Whether to normalize or denormalize the score column")
    args = parser.parse_args()
    main(args)

