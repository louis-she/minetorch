from minetorch import dataflow, option
from minetorch.processors import Processor
import cv2

@dataflow('Image Resizer', 'Resize image to any shape')
@option('column', help='Specified the column of the image to resize', default=9)
@option('size', help='Square size of the image')
class ImageResizer(Processor):
    def __init__(self, size):
        self.size = size

    def __call__(self, column, size):
        return cv2.resize(data[0], self.size)
