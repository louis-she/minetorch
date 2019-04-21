from pathlib import Path

import cv2
from torchvision import transforms
import inspect


class ProcessorBundler:

    def __init__(self, processors=[]):
        self.processors = processors

    def add_processors(self, processors=[]):
        self.processors += processors

    def prepend_processors(self, processors=[]):
        self.processors = processors + self.processors

    def __call__(self, data):
        for (processor, columns) in self.processors:
            if not isinstance(columns, tuple):
                columns = (columns,)

            for column in columns:
                data[column] = processor(data[column])
        return data


class Processor:

    def __call__(self):
        raise NotImplementedError()

    def source_file(self):
        return inspect.getsourcefile(self.__class__)

    def source_lineno(self):
        return inspect.getsourcelines(self.__class__)[1]


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


class Augmentor(Processor):
    """This is a tiny wrapper of albumentations
    see: https://github.com/albu/albumentations
    """

    def __init__(self, aug):
        """
        Args:
            aug (albumentations.core.composition.Compose)
        """
        self.aug = aug

    def __call__(self, tensor):
        return self.aug(image=tensor)['image']


class ImagenetNormalizor(Processor):

    def __init__(self):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, tensor):
        return self.normalize(tensor)
