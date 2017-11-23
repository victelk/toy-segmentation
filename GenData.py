import numpy as np
import cv2
import os.path

'''
Generate gray image with random brightness pixels.
Calculate average brightness of the pixels.
Label each pixel with brigness bigger than the average as 255,
and 0 otherwise.
'''


class GenData:
    def __init__(self, width, height):
        self.height = height  # image height
        self.width = width  # image width

    # Generate num images:
    def generate(self, file_prefix, num):
        width = self.width
        height = self.height
        for k in range(num):
            img = np.random.randint(256, size=(width, height))
            img_labels = np.zeros((width, height))
            img_labels[img > img.mean()] = 255
            dir_name = "data"
            image_name = file_prefix + "_" + "image" + str(k) + ".jpg"
            cv2.imwrite(os.path.join(dir_name, image_name), img)
            image_labels_name = file_prefix + "_" + "label_image" + str(k) + ".jpg"
            cv2.imwrite(os.path.join(dir_name, image_labels_name), img_labels)


if __name__ == "__main__":
    gd = GenData(6, 6)
    gd.generate("train", 100)
    gd.generate("test", 50)
    gd.generate("predict", 2)
