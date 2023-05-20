import os
import json
import threading
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Score.CalcAngle import get_score
import matplotlib.pyplot as plt
from collections import deque
from model import HighResolutionNet
from draw_utils import draw_keypoints
from Score.draw_line import  drawLine
import transforms
from Score.CalcAngle import CalcFinalScore
from DNN.dnn_infer import infer_dnn
from DNN.Dnn import Net


class flut():
    def __init__(self):
        self.queue =deque()
    def push(self,x):
        if len(self.queue) < 7:
            self.queue.append(x)
        else:
            self.queue.popleft()
            self.queue.append(x)
    def get_score(self):
        return sum(self.queue)/len(self.queue)


std_score = 0

if __name__ == '__main__':
    fcap = cv2.VideoCapture(r'../res/squats_video_01.mp4')
    success, frame = fcap.read()
    width = int(fcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(fcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    flip_test = True
    resize_hw = (256, 192)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r"../res/cxk_after.avi", fourcc, 30, (width, height))

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

    model_dnn = Net(4, 20, 1)
    model_dnn.load_state_dict(torch.load('./DNN/model.pth'))
    Confidence = []
    cnt = 0
    up = 1
    avg_win = flut()
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
            # print("infer cost: ",end-start,"s")
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
            score,y_label = infer_dnn(lAngle,model_dnn)

            # print("lAngle",lAngle)
            # score = Score.CalcAngle.get_score(lAngle)
            # print("y_label",y_label,"Score",score)
            avg_win.push(score)
            avg_score = avg_win.get_score()
            Confidence.append(avg_score)
            if(up == 1 and avg_score < 0.2):
                cnt += 1
                std_score = get_score(lAngle)
                up = 0
            if(up == 0 and avg_score >0.85):
                up = 1
            img = cv2.putText(img,"score:"+str(std_score),(400,150),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,0),5)
            img = cv2.putText(img,"cnt:"+str(cnt),(50,150),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
            #plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=7)
            plot_img = drawLine(img,keypoints)
            img = cv2.cvtColor(np.asarray(plot_img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (640,480),fx=1,fy=1)
            # cv2.imshow('img',img)
            out.write(img)
            # cv2.waitKey(1)
