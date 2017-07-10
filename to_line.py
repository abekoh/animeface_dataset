import os
import glob
import cv2
import numpy as np


class Anime2Contour():

    neiborhood24 = np.ones((5, 5), dtype=np.uint8)

    def __init__(self, src_dir_path, dst_dir_path):
        self.src_dir_path = os.path.normpath(src_dir_path)
        self.dst_dir_path = os.path.normpath(dst_dir_path)
        self._get_src_img_paths()

    def _get_src_img_paths(self):
        self.src_img_paths = list()
        for img_path in glob.glob(os.path.join(self.src_dir_path, '**/*.png'), recursive=True):
            self.src_img_paths.append(img_path)

    def run(self):
        for src_img_path in self.src_img_paths:
            gray_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            dilated = cv2.dilate(gray_img, Anime2Contour.neiborhood24, iterations=1)
            diff = cv2.absdiff(dilated, gray_img)
            contour = 255 - diff
            dst_img_path = self._to_dst_img_path(src_img_path)
            cv2.imwrite(dst_img_path, contour)
            print('saved {}'.format(dst_img_path))

    def _to_dst_img_path(self, src_img_path):
        dst_img_path = src_img_path.replace(self.src_dir_path + '/', '')
        dst_img_path = os.path.join(self.dst_dir_path, dst_img_path)
        if not os.path.exists(os.path.split(dst_img_path)[0]):
            os.makedirs(os.path.split(dst_img_path)[0])
        return dst_img_path


if __name__ == '__main__':
    a2c = Anime2Contour('../../animefaces', '../../animefaces_contour')
    a2c.run()
