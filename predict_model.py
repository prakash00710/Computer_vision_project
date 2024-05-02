from keras.models import load_model
from keras.preprocessing import image
from ImagePath import ImagePath
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F_torch
import cv2
class CarClassifier:
    out: any = -9999

    def __init__(self, img) -> None:
        self.img = img
    # def load(self):
            
    #         saved_model_path = r"C:\Users\PK\Downloads\car_classification_vscode\car_model_new.h5"
    #         loaded_model = load_model(saved_model_path)
    #         img = image.load_img(self.img, target_size=(224, 224))
    #         img_array = image.img_to_array(img)
    #         img_array = np.expand_dims(img_array, axis=0)
    #         prediction = loaded_model.predict(img_array)
    #         indices = [index for index, value in enumerate(prediction[0]) if value == 1]

    #         if indices == [0]:
    #             output =  ''
    #         elif indices == [1]:
    #             output =  'Audi'

    def load(self):
        img_path = self.img.lower()

        if img_path.endswith(('.jpg', '.jpeg')):
            pass  # No need to modify img_path
        else:
            # Convert image to JPEG format
            img_path = img_path.split('.')[0] + ".jpg"

            # Optional: You may want to save the converted image back to the same directory
            pil_img = Image.open(img_path)

            # Convert RGBA to RGB if necessary
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')

            if pil_img.mode == 'P':
                pil_img = pil_img.convert('RGB')

            pil_img.save(img_path)

        default_size = (128, 128)

        pil_img = Image.open(img_path)

        resized_img = F_torch.resize(pil_img, default_size)

        resized_img_np = np.array(resized_img)

        img1 = torch.from_numpy(resized_img_np.transpose(2, 0, 1)).float() / 255.0
        img1 = img1.unsqueeze(0)

        classModel = YOLO(r"C:\Users\PK\Downloads\car_classification_vscode\car_class.pt")
        classModel_1 = classModel.predict(source=img1, conf=0.1)

        names = classModel_1[0].names
        prediction = classModel_1[0].probs.top1
        predictionPercent = classModel_1[0].probs.top5conf.cpu().numpy()[0]
        finalAnswer = names[prediction]

        print()

        if predictionPercent > 0.75:
            finalAnswer = finalAnswer
        else:
            finalAnswer = 'Outside_curriculum'
        output=finalAnswer
        self.out = output
        return output