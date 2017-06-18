# -*- coding: utf-8 -*-
import cv2
import sys

class Video:
    def __init__(self, video_path):
        self.video_cap = cv2.VideoCapture(video_path)

    def get(self):
        ret, frame = self.video_cap.read()
        return frame


def detect(img, cascade_path='./lbpcascade_animeface.xml'):
    cascade = cv2.CascadeClassifier(cascade_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)

    face_ranges = cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))

    face_imgs = []

    for (x, y, w, h) in face_ranges:
        face_img = img[y : y + h, x : x + w]
        face_imgs.append(face_img)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    # cv2.imwrite('result.png', img)
    return face_imgs

def detect_video(video_path):
    src_video = Video(video_path)
    num = 0
    while 1:
        frame = src_video.get()
        face_imgs = detect(frame)
        for face_img in face_imgs:
            cv2.imwrite('pic/' + str(num) + '.png', face_img)
            num += 1


def debug():
    # img = cv2.imread('./cp.jpg')
    # detect(img)
    detect_video('./newgame_2.mp4')

if __name__ == '__main__':
    debug()
