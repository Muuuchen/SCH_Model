import csv
import os
import json
from Score.draw_line import drawLine
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from model import HighResolutionNet
from Score.CalcAngle import CalcFinalScore
from draw_utils import draw_keypoints
import transforms
file_name = r'../../datasets/squats_data/train/'
weights_path = "../../res/weights/model-209.pth"
def process_img(file_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()


    resize_hw = (256, 192)
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 这里应该对数据集迭代器中的每个元素

    with open('../res/future.csv', 'w') as file:
        writer =csv.writer(file)
        file.truncate(0)
        with torch.no_grad():
            for img_path_label in DataStream(file_name):
                img = cv2.imread(img_path_label[0])
                label = img_path_label[1]
                img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
                img_tensor = torch.unsqueeze(img_tensor, dim=0)
                start = time.time()
                outputs = model(img_tensor.to(device))
                end = time.time()
                print("infer cost: ",end-start)

                keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
                keypoints = np.squeeze(keypoints)
                scores = np.squeeze(scores)
                # print(keypoints,scores)
                try:
                    uAngle,lAngle =  CalcFinalScore(keypoints)
                except ValueError as e:
                    print(img_path_label)
                    print(e)

                # print(uAngle,lAngle)
                wData = lAngle + [label=='up']
                writer.writerow(wData)
                # img = drawLine(img,keypoints)
                # img = cv2.resize(img,(640,480),1,1)
                # cv2.imshow('img', img)
                # cv2.waitKey(20000)


# 直接指定用train下的数据和分类讨论构建两种数据类型 后续作处理

class DataStream:
    def __init__(self, file_name):
        self.file_name = file_name
        self.idx = 0
        self.data = []
        for dir in os.listdir(file_name):
            if dir == 'up':
                label = 'up'
            elif dir == 'down':
                label = 'down'
            imgs_path = os.path.join(file_name, dir)
            for img_name in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path, img_name)
                self.data.append((img_path, label))

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.data):
            raise StopIteration
        item = self.data[self.idx]
        self.idx += 1
        return item

if __name__ == "__main__":
    process_img(file_name)




