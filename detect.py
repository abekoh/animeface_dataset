import cv2
import os


class Video:

    def __init__(self, video_path):
        self.video_cap = cv2.VideoCapture(video_path)
        self.frame_i = 0

    def get(self):
        ret, frame = self.video_cap.read()
        self._skip()
        if ret:
            return frame, self.frame_i
        return None, self.frame_i

    def _skip(self, count_n=24):
        count_i, ret = 0, True
        while ret and count_i < count_n:
            ret, _ = self.video_cap.read()
            self.frame_i += 1
            count_i += 1


class Detector:

    def __init__(self, src_dir_path, dst_dir_path='dst', min_size=(256, 256), resized_size=(256, 256), cascade_path='./lbpcascade_animeface.xml', video_exts=['.mp4']):
        self.src_dir_path = os.path.normpath(src_dir_path)
        self.dst_dir_path = os.path.normpath(dst_dir_path)
        self.min_size = min_size
        self.resized_size = resized_size
        self.cascade_path = cascade_path
        self.video_exts = video_exts
        self.video_paths = []
        self._get_video_paths()

    def _get_video_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.src_dir_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.splitext(filepath)[1] in self.video_exts:
                    self.video_paths.append(filepath)

    def detect(self):
        for video_path in self.video_paths:
            print('detecting faces in {}'.format(video_path))
            dst_video_dir_path = self._get_dst_video_dir_path(video_path)
            video = Video(video_path)
            while 1:
                img, frame_i = video.get()
                if img is None:
                    break
                face_imgs = self._get_faces_one_frame(img)
                if len(face_imgs) > 0:
                    print('detect {} faces at {} frame'.format(len(face_imgs), frame_i))
                    for face_i, face_img in enumerate(face_imgs):
                        resized_face_img = self._resize_img(face_img, self.min_size)
                        cv2.imwrite(os.path.join(dst_video_dir_path, '{}_{}.png'.format(frame_i, face_i)), resized_face_img)

    def _get_dst_video_dir_path(self, video_path):
        if os.path.split(video_path)[0] == self.src_dir_path:
            dst_video_dir_path = self.dst_dir_path
        else:
            dirname = video_path.split('/')[-2]
            filename = video_path.split('/')[-1].split('.')[0]
            dst_video_dir_path = os.path.join(self.dst_dir_path, dirname, filename)
        if not os.path.exists(dst_video_dir_path):
            os.makedirs(dst_video_dir_path)
        return dst_video_dir_path

    def _get_faces_one_frame(self, img):
        cascade = cv2.CascadeClassifier(self.cascade_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.equalizeHist(gray_img)
        face_ranges = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size)
        face_imgs = []
        for x, y, w, h in face_ranges:
            face_img = img[y:y + h, x:x + w]
            face_imgs.append(face_img)
        return face_imgs

    def _resize_img(self, img, size):
        return cv2.resize(img, size)


if __name__ == '__main__':
    detector = Detector('src')
    detector.detect()
