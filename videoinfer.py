import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import Score.CalcAngle
from model import HighResolutionNet
from draw_utils import draw_keypoints
from Score.draw_line import  drawLine
import transforms
from KNN_cnt.knn_cnt import myKNN
from Score.CalcAngle import CalcFinalScore


if __name__ == '__main__':
    fcap = cv2.VideoCapture(r'../res/squats_video_01.mp4')
    success, frame = fcap.read()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    flip_test = True
    resize_hw = (256, 192)
    weights_path = "../res/weights/model-209.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)
        # create model
    model = HighResolutionNet()

    # 载入你自己训练好的模型权重
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()


    #KNN模块
    knn = myKNN()
    knn.train_knn()

    while success:
        success, frame = fcap.read()
        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        with torch.no_grad():
            start = time.time()
            outputs = model(img_tensor.to(device))
            end = time.time()
            print("infer cost: ",end-start,"s")
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
                )
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            # scores = np.squeeze(scores)
            uAngle, lAngle = CalcFinalScore(keypoints) #获得角度参数
            # print(lAngle)
            y_label = knn.predict_knn([lAngle])
            score = Score.CalcAngle.get_score(lAngle)
            print("y_label",y_label,"Score",score)


            #plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=7)
            plot_img = drawLine(img,keypoints)
            img = cv2.cvtColor(np.asarray(plot_img), cv2.COLOR_RGB2BGR)
            # cv2.imshow('img',img)
            # cv2.waitKey(1)