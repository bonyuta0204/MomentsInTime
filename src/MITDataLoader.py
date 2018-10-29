import os
import numpy as np
import pandas as pd
from MITData import MITData

class MITDataLoader:
    def __init__(self, root_dir="../data/MIT_data", train=True):
        # self.train_index = read_csv(csffile)
        # self.test_index = read_csv(csffile)
        self.train = train
        self.root_dir = root_dir

        if self.train:
            self.data_dir = os.path.join(root_dir, "training")
        else:
            self.data_dir = os.path.join(root_dir, "test")

        self.index = pd.read_csv(os.path.join(self.root_dir, "train_index.csv"),
                                    index_col="index")
        self._i = 0

    # iter data set based on self.index
    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.index.shape[0]:
            raise StopIteration
        index = self._i
        data = self.index.loc[self._i]
        mitdata = MITData(data["directory"], data["filename"], data["train"])
        self._i  += 1
        return index, mitdata

    def __getitem__(self, x):
        data = self.index.loc[x]
        mitdata = MITData(data["directory"], data["filename"], data["train"])
        return mitdata

    # create csv file based on directory
    def parse_directory(self):
        index = []
        # for data in directory
        # index.append({"directory": directory, "train":train, "filename": video,  "category": category})
        n = 0
        for category in os.listdir(self.data_dir):
            for movie in os.listdir(os.path.join(self.data_dir, category)):
                index.append({"index": n, 
                            "directory": os.path.join(self.data_dir, category), 
                            "filename":movie, "train":"train", "category":category})
                n  += 1;

        df = pd.DataFrame(index)
        df = df.set_index("index")
        df.to_csv(os.path.join(self.root_dir, "train_index.csv"), index_label="index")

if __name__ == '__main__':
    train_loader = MITDataLoader(train=True)
    data = train_loader[247]
    data.crop_image()
