import os
import sys
import argparse
import requests
from tqdm import tqdm

url = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
tar_file_name = "clean_midi.tar.gz"

def download(data_dir):
    if os.path.exists(data_dir):
        print("Error: Directory already exists")
        return
    os.mkdir(data_dir)
    os.chdir(data_dir)
    print("Downloading from {}".format(url))
    response = requests.get(url, stream=True)
    with open(tar_file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to place the dataset")
    parser.add_argument("--flat", action="store_true" ,help="whether to place all midi in one dir")
    args = parser.parse_args()
    download(args.path)
