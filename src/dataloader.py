import os

import pandas as pd
from mitdata import MITData

ROOT_DIR = os.path.join("/", *os.path.abspath(__file__).split("/")[:-2])


class MITDataLoader:
    def __init__(self,
                 root_dir=os.path.join(ROOT_DIR, "data/MIT_data"),
                 train=True):
        self.train = train
        self.root_dir = root_dir

        if self.train:
            self.data_dir = os.path.join(root_dir, "training")
        else:
            self.data_dir = os.path.join(root_dir, "test")
        try:
            if self.train:
                self.index = pd.read_csv(
                    os.path.join(self.root_dir, "train_index.csv"),
                    index_col="index")
            else:
                self.index = pd.read_csv(
                    os.path.join(self.root_dir, "test_index.csv"),
                    index_col="index")
        except FileNotFoundError:
            print("parseing directories")
            self.parse_directory()
            self.index = pd.read_csv(
                os.path.join(self.root_dir, "test_index.csv"),
                index_col="index")

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.index.shape[0]:
            print("end of iteration")
            raise StopIteration
        data = self.index.loc[self._i]
        mitdata = MITData(self._i, data["directory"], data["filename"],
                          data["train"], data["object_label"],
                          data["scene_label"])
        self._i += 1
        return mitdata

    def __getitem__(self, x):
        data = self.index.loc[x]
        mitdata = MITData(self._i, data["directory"], data["filename"],
                          data["train"], data["object_label"],
                          data["scene_label"])
        return mitdata

    def reset(self):
        self._i = 0

    def save_index(self):
        if self.train:
            self.index.to_csv(
                os.path.join(self.root_dir, "train_index.csv"),
                index_label="index")
        else:
            self.index.to_csv(
                os.path.join(self.root_dir, "test_index.csv"),
                index_label="index")

    def filter_data(self, filter_func):
        """
        return iterator filtered by filter_func
        filter_func takes one MITData as args

        Example:
            for data in train_loader.filter_data(lambda data: data.index <= 200):
                print(data.index)
        will print number 1 ~ 200
        """
        return filter(filter_func, iter(self))


    # create csv file based on directory
    def parse_directory(self):
        index = []
        # for data in directory
        # index.append({"directory": directory,
        # "train": train, "filename": video,
        # "category": category})
        n = 0
        if self.train:
            for category in os.listdir(self.data_dir):
                for movie in os.listdir(os.path.join(self.data_dir, category)):
                    index.append({
                        "index":
                        n,
                        "directory":
                        os.path.join(self.data_dir, category),
                        "filename":
                        movie,
                        "train":
                        "train",
                        "category":
                        category
                    })
                    n += 1
        else:
            for category in os.listdir(self.data_dir):
                for movie in os.listdir(os.path.join(self.data_dir, category)):
                    index.append({
                        "index":
                        n,
                        "directory":
                        os.path.join(self.data_dir, category),
                        "filename":
                        movie,
                        "train":
                        "test",
                        "category":
                        category
                    })
                    n += 1

        df = pd.DataFrame(index)
        df = df.set_index("index")
        if self.train:
            df.to_csv(
                os.path.join(self.root_dir, "train_index.csv"),
                index_label="index")
        else:
            df.to_csv(
                os.path.join(self.root_dir, "test_index.csv"),
                index_label="index")


if __name__ == '__main__':
    train_loader = MITDataLoader(train=True)
    for data in train_loader.filter_data(lambda data: data.index <= 200):
        print(data.object_label)
        print(data.object_label_list)
