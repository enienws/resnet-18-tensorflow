from .kinetics import Kinetics
import os
import logging
import itertools
import cv2
import pickle

LOGGER = logging.getLogger(__name__)

class KineticsClustered(Kinetics):
    def __init__(self, num_set, base_path, shuffle=False, num_frames=1, skips=(0,)):
        self.num_set = num_set
        super().__init__(num_set, base_path, shuffle, num_frames, skips)

    def get_filename(self, name):
        if name not in self.keys:
            raise KeyError('not exists name at %s' % name)
        LOGGER.debug('[Kinetics.get] %s', name)
        filename_label = os.path.join(self.base_path, 'centroids', name + '.label')
        filename_image = os.path.join(self.base_path, 'processed', name + '.mp4')
        exists = os.path.exists(filename_image)
        labels_existance = os.path.exists(filename_label)
        if exists and not labels_existance:
            print("Video exists but labels are not: {}".format(name))
            exists = False
        return exists, filename_label, filename_image


    def __iter__(self, num_frames=None, skips=None):
        num_frames = num_frames if num_frames is not None else self.num_frames
        skips = skips if skips is not None else self.skips

        for name in self.names:
            exists, filename_label, filename_image = self.get_filename(name)
            if not exists:
                continue
            print("Current video [{}]: {}".format(self.keys.index(name), filename_label))
            index = -1
            images = []
            labels = []
            capture = cv2.VideoCapture(filename_image)
            with open(filename_label, 'rb') as f:
                labels_obj = pickle.load(f)
            label_index = -1
            for _, skip in itertools.cycle(enumerate(skips)):
                if len(images) == num_frames:
                    images = []
                    labels = []
                for _ in range(skip):
                    capture.read()
                    label_index = label_index + 1
                ret, image = capture.read()
                label_index = label_index + 1
                if not ret:
                    break
                image = cv2.resize(image, (256, 256))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(256, 256, 1)
                # ret, image = True, None

                # if label_index >= len(labels_obj):
                #     ret = False

                images.append(image)
                labels.append(labels_obj[label_index])
                if len(images) == num_frames and len(labels) == num_frames:
                    index += 1
                    yield [index, images, labels]