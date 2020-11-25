import os
import sys
import argparse
import requests
import tarfile

from tqdm import tqdm

url = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"
tar_file_name = "clean_midi.tar.gz"

def download():
    print("Downloading from {}".format(url))
    head = requests.head(url)
    response = requests.get(url, stream=True)
    print("Total size:" , int(head.headers['Content-length']) // 1024)
    with open(tar_file_name, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)

def extract():
    print("Extracting ...")
    tar = tarfile.open(tar_file_name, mode = "r:gz")
    index = 0
    for midi in tqdm(tar.getmembers()):
        if not midi.name.endswith(".mid"):
            continue
        midi.name = "{}.mid".format(index)
        tar.extract(midi)
        index += 1
    tar.close()

def remove_tar():
    os.remove(tar_file_name)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Path to place the dataset")
    args = parser.parse_args()
    if os.path.exists(args.path):
        print("Error: Directory already exists")
        exit(0)
    os.mkdir(args.path)
    os.chdir("dataset")
    download()
    extract()
    remove_tar()
    print("Done!")
