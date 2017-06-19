# -*- coding: utf-8 -*-
import cv2
import os


class Video:

    def __init__(self, video_path):
        self.video_cap = cv2.VideoCapture(video_path)
        self.frame_i = 0

    def get(self):
        ret, frame = self.video_cap.read()
        self.frame_i += 1
        self._skip()
        if ret:
            return frame, self.frame_i
        return None, self.frame_i

    def _skip(self, count_n=24 * 5):
        count_i, ret = 0, True
        while ret and count_i < count_n:
            ret, _ = self.video_cap.read()
            self.frame_i += 1
            count_i += 1


class Detector:

    def __init__(self, video_path, min_size=(128, 128), cascade_path='./lbpcascade_animeface.xml', output_dir_path='output'):
        self.video = Video(video_path)
        self.min_size = min_size
        self.cascade_path = cascade_path
        self.output_dir_path = output_dir_path
        self.output_video_dir_path = os.path.join(output_dir_path, os.path.basename(video_path))
        if not os.path.exists(self.output_video_dir_path):
            os.makedirs(self.output_video_dir_path)

    def get_faces_one_frame(self, img):
        cascade = cv2.CascadeClassifier(self.cascade_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)
        face_ranges = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size)
        face_imgs = []
        for x, y, w, h in face_ranges:
            face_img = img[y:y + h, x:x + w]
            face_imgs.append(face_img)
        return face_imgs

    def save_img(self, img, frame_i, face_i):
        cv2.imwrite(os.path.join(self.output_video_dir_path, '{}_{}.png'.format(frame_i, face_i)), img)

    def save_faces(self):
        while 1:
            frame, frame_i = self.video.get()
            if frame is None:
                break
            face_imgs = self.get_faces_one_frame(frame)
            for face_i, face_img in enumerate(face_imgs):
                self.save_img(face_img, frame_i, face_i)


if __name__ == '__main__':
    detector = Detector('./eupho1_1.mp4')
    detector.save_faces()
