import cv2

class ImageProcess():
    @staticmethod
    def resize(size=(256,256)):
        def _resize(images):
            for idx, image in enumerate(images):
                img_resized = cv2.resize(image, size)
            return images
        return _resize