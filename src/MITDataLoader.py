import os
import numpy as np

class MITDataLoader:
    def __init__(self):
        # self.train_index = read_csv(csffile)
        # self.test_index = read_csv(csffile)

    # iter data set based on self.index
    def __iter__(self):
        pass

    def __next__(self):
        pass

    # create csv file based on directory
    def parse_directory(self, options):
        index = []
        # for data in directory
        index.append({"directory": directory, "train":train, "filename": video,  "category": category})


