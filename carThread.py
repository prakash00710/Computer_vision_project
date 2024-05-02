import os
from ImagePath import ImagePath
from predict_model import CarClassifier

class carThread():
    out = {}

    def __init__(self) -> None:
        pass

    def load(self, imgUrl:str):
        imgPath = ImagePath(imgUrl)
        model = CarClassifier(imgPath.fileNameWPath)

        model.load()

        if (model.out != -9999):
            self.out['Car Class'] = model.out

        return self.out

