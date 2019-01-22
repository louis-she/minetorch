from pathlib import Path

import cv2
from torchvision import transforms


class ProcessorBundler:

    def __init__(self, processors=[]):
        self.processors = processors

    def register_processor(self, processor, column):
        self.processors.append((processor, column))

    def __call__(self, data):
        for (processor, column) in self.processors:
            data[column] = processor(data[column])
        return data


class Processor:

    def __call__(self):
        raise NotImplementedError()


class PathMaker(Processor):
    def __init__(self, base_directory, extname=None):
        self.base_directory = Path(base_directory)
        self.extname = extname

    def __call__(self, filename):
        path = self.base_directory / filename
        if self.extname is not None:
            path = path.with_suffix(self.extname)
        return path.as_posix()


class ImageLoader(Processor):

    def __init__(self, image_size=None):
        self.image_size = image_size

    def __call__(self, filename):
        image = cv2.imread(filename)
        if self.image_size is not None:
            image = cv2.resize(image, self.image_size)
        return image


class ImageAgumentor(Processor):

    def __init__(self, image_augment_instance):
        self.image_augment_instance = image_augment_instance

    def __call__(self, image):
        return self.image_augment_instance.augment(image)


class ImagenetNormalizor(Processor):

    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, tensor):
        return self.normalize(tensor)
