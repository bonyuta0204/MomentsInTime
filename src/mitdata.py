import os

from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np

class MITData:
    def __init__(self, index, dir_path, filename, train, object_label,
                 scene_label):
        self.index = index
        self.dir_path = dir_path
        self.filename = filename
        self.moviepath = os.path.join(self.dir_path, self.filename)
        self.train = train
        self.object_label = object_label
        self.object_label_list = str(self.object_label).split(" ")
        self.scene_label = scene_label
        self.category = os.path.split(self.dir_path)[1]
        self.image_dir = os.path.join(self.dir_path,
                                      self.filename.split(".")[0])
        self.converted_moviepath = os.path.join("../data/MIT_data/h264",
                                                *self.moviepath.split("/")[3:])

    def has_object_label(self, label):
        if label in self.object_label_list:
            return True
        else:
            return False

    def crop_image(self, frames=[22, 45, 67]):
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)

        infile = os.path.join(self.dir_path, self.filename)

        video = []
        cap = cv2.VideoCapture(infile)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                video.append(frame)
            else:
                break
        cap.release()

        for frame in frames:
            img_path = os.path.join(self.image_dir,
                                    "{0:04d}.png".format(frame))
            if os.path.isfile(img_path):
                continue
            img = video[frame]
            img = Image.fromarray(np.uint8(img))
            img.save(img_path)

    def load_images(self, frames=[22, 45, 67], type="imagenet"):
        paths = []
        for frame in frames:
            img_path = os.path.join(self.image_dir,
                                    "{0:04d}.png".format(frame))
            paths.append(img_path)

        return paths

    def show_images(self):
        images = self.load_images()
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images)
        for i in range(n_images):
            img = np.asarray(Image.open(images[i]))
            axes[i].imshow(img)
            axes[i].axis('off')
        return fig, axes


if __name__ == "__main__":
    data = MITData("../data/MIT_data/training/aiming/", "v035_0005.mp4",
                   "training")
    print(data.object_label_list)
