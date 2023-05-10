import numpy
import numpy as np


def get_cos(e1, e2):
    '''
    Args:
        e1: array 假定都是正方向都从中点向外
        e2: 假定都是正方向都从中点向外
    Returns: cos 角度
    '''

    cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    angle = np.arccos(cos_angle)
    return angle

class Body:
    def __init__(self, upper, keypoint):
        if(upper):
            self.Edges = {'1':[7,9],'2':[5,7],'3':[5,6],'4':[6,8],'5':[8,10]}
        else:
            self.Edges = {'1':[13,15],'2':[11,13],'3':[11,12],'4':[12,14],'5':[14,16]}

        self.Vector =[]
        #计算各组向量
        for i in range(5):
            self.Vector.append([keypoint[self.Edges[str(i+1)][0]]-keypoint[self.Edges[str(i+1)][0]],keypoint[self.Edges[str(i+1)][1]]-keypoint[self.Edges[str(i+1)][1]]])
            #v 5-7
        self.A = get_cos(np.array(self.Vector[0]), -np.array(self.Vector[1]))
        self.B = get_cos(np.array(self.Vector[1]), np.array(self.Vector[2]))
        self.C = get_cos(-np.array(self.Vector[2]), np.array(self.Vector[3]))
        self.D = get_cos(-np.array(self.Vector[3]), -np.array(self.Vector[4]))
    def getEdges(self):
        return self.Edges
    def getVector(self):
        return self.Vector
    def getAngle(self):
        return [self.A,self.B,self.C,self.D]



if __name__ == "__main__":
    #
    keypoint = [[]]
    Upper_body = Body(1,keypoint=keypoint)
    Lower_body = Body(0,keypoint=keypoint)
    print(Upper_body.getEdges())
    print(Upper_body.getVector())
    print(Upper_body.getAngle())
    print("lower")
    print(Lower_body.getEdges())
    print(Lower_body.getVector())
    print(Lower_body.getAngle())