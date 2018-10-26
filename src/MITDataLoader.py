import os
import numpy as np
import pandas as pd

class MITDataLoader:
    def __init__(self, train=True):
        # self.train_index = read_csv(csffile)
        # self.test_index = read_csv(csffile)
        self.train = train
        if self.train:
            self.root_dir = "../data/MIT_exp/training"

    # iter data set based on self.index
    def __iter__(self):
        pass

    def __next__(self):
        pass

    # create csv file based on directory
    def parse_directory(self):
        index = []
        # for data in directory
        # index.append({"directory": directory, "train":train, "filename": video,  "category": category})
        n = 0
        for category in os.listdir(self.root_dir):
            for movie in os.listdir(os.path.join(self.root_dir, category)):
                index.append({"index": n, 
                            "directory": os.path.join(self.root_dir, category), 
                            "filename":movie, "train":"train", "category":category})
                n  += 1;

        df = pd.DataFrame(index)
        df = df.set_index("index")
        df.to_csv(os.path.join("train_index.csv"), index_label="index")

if __name__ == '__main__':
    train_loader = MITDataLoader(train=True)
    train_loader.parse_directory()

