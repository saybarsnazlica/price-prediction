import argparse
import json
import pickle

import pandas as pd
import requests

from utils import process_data, transform_test_data

URL = "http://127.0.0.1:5000//predict"

parser = argparse.ArgumentParser(
    description="Use a row from the test data table as a request"
)
parser.add_argument("-r", "--row", type=int, required=True, help="The row index to use as a request, starting from 0")
args = parser.parse_args()


def prepare_data(row, test_data, label_binarizers, count_vectorizers):
    """Create a slice (single sample) from the test data to make a prediction

    Args:
        row (int): Row index starting from 0

    Returns:
        list: Test features
    """
    if row >= test_data.shape[0]:
        raise Exception(f"Row index must be less than {test_data.shape[0]}")
    elif row < 0:
        raise Exception("Row index must be 0 or greater")
    
    test_single_sample = test_data.iloc[row : row + 1, :]
    test_proc = process_data(test_single_sample.drop(["id", "seller_id"], axis=1))
    X_test = transform_test_data(test_proc, label_binarizers, count_vectorizers)

    # Print the input features for a sanity check
    print("In: ")
    print(test_single_sample.to_json(orient="records"))

    return X_test.toarray().tolist()


def main():
    label_bin = pickle.load(open("models/label_binarizers", "rb"))
    count_vec = pickle.load(open("models/count_vectorizers", "rb"))
    test = pd.read_csv("data/mercari_test.csv.gz", compression="gzip")
    data = prepare_data(args.row, test, label_bin, count_vec)

    # Serialize the data into json and send the request to the model
    payload = {"query": json.dumps(data)}
    prediction = requests.post(URL, data=payload).json()
    print("Out:")
    print(prediction)


if __name__ == "__main__":
    main()
