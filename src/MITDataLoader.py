import os
import numpy as np

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
                index.append({"index": n, "directory": os.path.join(self.root_dir, category), "filename":movie, "train":"train", "category":category})
                n  += 1;
        return index

if __name__ == '__main__':
    import pandas as pd
    train_loader = MITDataLoader(train=True)
    df = pd.DataFrame(train_loader.parse_directory())
    df.to_csv("train_index.csv")

