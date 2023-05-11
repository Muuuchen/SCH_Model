import numpy as np
from std_angle import lower_Angle_std, upper_Angle_std


def get_cos(e1, e2):
    """
    Args:
        e1: array 假定都是正方向都从中点向外
        e2: 假定都是正方向都从中点向外
    Returns: cos 角度
    """

    cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    angle = np.arccos(cos_angle)
    return angle


def get_score(angle_upper, angle_lower):
    alpha = 5.0  # 这是一个神奇的超参，你可以根据它构造出你想要的分数
    MSE = 0
    for i in range(len(angle_lower)):
        MSE += (angle_upper[i] - upper_Angle_std[i]) ** 2
        MSE += (angle_lower[i] - lower_Angle_std[i]) ** 2
    return 100 - (MSE / 8) * alpha


class Body:
    def __init__(self, upper, keypoint):
        if (upper):
            self.Edges = {'1': [7, 9], '2': [5, 7], '3': [5, 6], '4': [6, 8], '5': [8, 10]}
        else:
            self.Edges = {'1': [13, 15], '2': [11, 13], '3': [11, 12], '4': [12, 14], '5': [14, 16]}

        self.Vector = []
        # 计算各组向量
        for i in range(5):
            x = keypoint[self.Edges[str(i + 1)][0]][0] - keypoint[self.Edges[str(i + 1)][1]][0]
            y = keypoint[self.Edges[str(i + 1)][0]][1] - keypoint[self.Edges[str(i + 1)][1]][1]
            self.Vector.append([x, y])
        self.A = get_cos(np.array(self.Vector[0]), -np.array(self.Vector[1]))
        self.B = get_cos(np.array(self.Vector[1]), np.array(self.Vector[2]))
        self.C = get_cos(-np.array(self.Vector[2]), np.array(self.Vector[3]))
        self.D = get_cos(-np.array(self.Vector[3]), -np.array(self.Vector[4]))

    def getEdges(self):
        return self.Edges

    def getVector(self):
        return self.Vector

    def getAngle(self):
        return [self.A, self.B, self.C, self.D]


def CalcFinalScore(keypoint):
    Upper_body = Body(1, keypoint=keypoint)
    Lower_body = Body(0, keypoint=keypoint)
    print("upper")
    print("Edges", Upper_body.getEdges())
    print("Vector", Upper_body.getVector())
    print("Angle", Upper_body.getAngle())
    print("lower")
    print("Edges", Lower_body.getEdges())
    print("Vector", Lower_body.getVector())
    print("Angle", Lower_body.getAngle())
    print("SCORE", get_score(Upper_body.getAngle(), Lower_body.getAngle()))


if __name__ == "__main__":
    #
    keypoint = [[116.68128, 72.93137],
                [123.47448, 59.362743],
                [103.094894, 66.14706],
                [137.06087, 66.14706],
                [75.92212, 72.93137],
                [150.64725, 140.7745],
                [48.749344, 147.55882],
                [171.02682, 222.18628],
                [48.749344, 262.89215],
                [96.301704, 195.04903],
                [69.12893, 357.87256],
                [137.06087, 330.7353],
                [62.33573, 330.7353],
                [157.44044, 473.20587],
                [69.13893, 479.9902],
                [82.71532, 574.9706],
                [75.92212, 622.46075]]
    # 这个数据是主目录下那个戴眼镜的男的
    CalcFinalScore(keypoint)
