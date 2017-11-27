import numpy as np
import cv2
import os.path
import csv

'''
Generate gray image with random brightness pixels.
Calculate average brightness of the pixels.
Label each image by its average brightness.
'''


class GenData:
    def __init__(self, width, height):
        self.height = height  # image height
        self.width = width  # image width

    # Generate num images:
    def generate(self, file_prefix, num):
        dir_name = "data"
        label_file_name = file_prefix + "_" + "labels" + ".csv"
        f=open(os.path.join(dir_name, label_file_name),"w")
        width = self.width
        height = self.height
        all_mean=np.array([])
        for k in range(num):
            img = np.random.randint(256, size=(width, height))
            # img_labels = np.zeros((width, height))
            # img_labels[img > img.mean()] = 255
            image_name = file_prefix + "_" + "image" + str(k).zfill(3) + ".png"
            cv2.imwrite(os.path.join(dir_name, image_name), img)
            f.write(str(img.mean()) + "\n")
            all_mean=np.append(all_mean,img.mean())
            # image_labels_name = file_prefix + "_" + "label_image" + str(k) + ".png"
            # cv2.imwrite(os.path.join(dir_name, image_labels_name), img_labels)
        f.close()
        print(all_mean.mean())


if __name__ == "__main__":
    gd = GenData(6, 6)
    gd.generate("train", 10000)
    gd.generate("test", 1000)
    gd.generate("predict", 10)
