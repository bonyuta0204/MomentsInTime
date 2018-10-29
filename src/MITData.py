import os
from PIL import Image
import cv2
import numpy as np
import torch

class MITData:
    def __init__(self, dir_path, filename, train):
        self.dir_path = dir_path
        self.filename = filename
        self.train = train
        self.image_dir = os.path.join(self.dir_path, self.filename.split(".")[0])

    def crop_image(self, frames=[22, 45, 67]):
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)

        infile = os.path.join(self.dir_path, self.filename)

        video = []
        cap = cv2.VideoCapture(infile)
        while(cap.isOpened()):
            ret,  frame = cap.read()
            if ret:
                video.append(frame)
            else:
                break
        cap.release()

        for frame in frames:
            img_path = os.path.join(self.image_dir, "{0:04d}.png".format(frame))
            if os.path.isfile(img_path):
                continue
            img = video[frame]
            img = Image.fromarray(np.uint8(img))
            img.save(img_path)

    def image_tensor(self, frames=[22, 45, 67]):
        images = []
        for frame in frames:
            img_path = os.path.join(self.image_dir, "{0:04d}.png".format(frame))
            img = Image.open(img_path)
            img_array = np.asarray(img)
            images.append(img_array.transpose(2, 0, 1))
        images = np.array(images)
        return torch.from_numpy(images)

if __name__ == "__main__":
    data = MITData("../data/MIT_data/training/aiming/", "v035_0005.mp4", "training")
    print(data.image_dir)
    data.crop_image()
    data.image_tensor()
