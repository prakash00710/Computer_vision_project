from io import BytesIO
import cv2
import os
from PIL import Image

class ImagePath:
    def __init__(self, image_url):
        self.image_url = image_url
        self.fileName = os.path.basename(image_url)
        self.local()

    def local(self):
        local_directory = r"C:\Users\PK\Downloads\car_classification_vscode\Cars Dataset"
        self.fileNameWPath = os.path.join(local_directory, self.fileName)
        self.cv2Img = cv2.imread(self.fileNameWPath)

