import torch
from DNN.Dnn import Net

# net = Net(4,20,1)
# 加载模型


def infer_dnn(data, model_dnn):
    data = torch.tensor(data,dtype=torch.float32)

    # print(data)
    y_pred = model_dnn(data)

    y_pred = torch.sigmoid(y_pred)
    y_pred_test_class = torch.round(y_pred)
    return y_pred.item(),y_pred_test_class.item()

