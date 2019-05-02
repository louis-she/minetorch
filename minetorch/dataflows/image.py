from minetorch import dataflow, option
from minetorch.processors import Processor
import cv2


@dataflow('Image Resizer')
@option('column', help='Specified the column of the image to resize', default='0')
@option('size', help='Square size of the image')
class ImageResizer(Processor):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        return cv2.resize(data[0], self.size)
