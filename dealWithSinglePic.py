import os
import json
from Score.CalcAngle import get_score
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms
from Score.draw_line import  drawLine
from Score.CalcAngle import CalcFinalScore
from DNN.dnn_infer import infer_dnn
from DNN.Dnn import Net


def predict_all_person():
    # TODO
    pass


def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = True
    resize_hw = (256, 192)
    img_path = "../res/t3.png"
    weights_path = "../res/weights/model-209.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    model_dnn = Net(4, 20, 1)
    model_dnn.load_state_dict(torch.load('./DNN/model.pth'))



    with torch.no_grad():
        start = time.time()
        outputs = model(img_tensor.to(device))
        end = time.time()
        print("infer cost: ",end-start)
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
        scores = np.squeeze(scores)
        uAngle, lAngle = CalcFinalScore(keypoints)
        score, y_label = infer_dnn(lAngle, model_dnn)
        std_score = get_score(lAngle)
        score = score if y_label else 1-score
        state = "up" if y_label == 1 else "down"
        img = cv2.putText(img, "Confidence:" + "{:.3f}".format(score), (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (250, 0, 0), 2)
        img = cv2.putText(img, "state:" + str(state), (0, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0, 0, 255), 2)
        img = cv2.putText(img, "std_score:" + "{:.3f}".format(std_score), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        plot_img = drawLine(img, keypoints)
        #plt.imshow(plot_img)
        #plt.show()
        cv2.imwrite(f"res/test_result3.jpg",plot_img)


if __name__ == '__main__':
    predict_single_person()
