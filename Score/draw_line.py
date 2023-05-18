import cv2
import numpy as np
import random
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
def drawLine(img, keypoint):

   edges=[[3,1],[1,0],[2,0],[4,2],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12]
          ,[11,12],[11,13],[13,15],[12,14],[14,16]]
   extra_centor_point  = [(keypoint[5][0] + keypoint[6][0])/2, (keypoint[5][1] + keypoint[6][1])/2]
   print(extra_centor_point)
   color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
   for edge in edges:
       start_point =list(map(int,keypoint[edge[0]]))
       end_point = list(map(int,keypoint[edge[1]]))
       cv2.line(img, tuple(start_point),tuple(end_point),color,thickness=5)
   cv2.line(img, (int(keypoint[0][0]), int(keypoint[0][1])),(int(extra_centor_point[0]), int(extra_centor_point[1])),color,thickness=5 )
   return img

if __name__ == "__main__":
    img = np.zeros((512,512,3), np.uint8)
    drawLine(img, keypoint)
    cv2.imshow("img",img)
    cv2.waitKey()