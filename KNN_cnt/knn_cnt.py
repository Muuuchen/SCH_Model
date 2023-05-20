import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
from sklearn.model_selection  import cross_val_score
# 打开CSV文件并读取数据
file_path = '/home/muuuchen/Desktop/HRNet/SCH_Model/res/future.csv'
def get_data(file_path):
    with open(file_path, newline='') as file:
        reader = csv.reader(file)
        data_x = []
        data_y = []
        for list in reader:
            data_x.append([float(x) for x in list[0:-1]])
            data_y.append(1 if list[-1]=="True" else 0)
            #1 up  #     0 down
        return data_x,data_y
    # 输出数据


class myKNN():
    def __init__(self):
        x_data,y_data = get_data(file_path)
        self.knn = knn = KNeighborsClassifier(n_neighbors=2)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=0.2)

    def train_knn(self):
        self.knn.fit(self.x_train,self.y_train)

    def predict_knn(self,x):
        y_predict = self.knn.predict(x)

        return y_predict



if __name__ == "__main__":
    x_data, y_data = get_data(file_path)
    k_error = []
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    # for k in range(1,10):
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knn, x_data, y_data, cv=6, scoring='accuracy')
    #     k_error.append(1 - scores.mean())
    # 根据kerror看出k取2的时候误差最小
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    percent = knn.score(x_test, y_test)
    k_nearest_indices, k_nearest_distances = knn.kneighbors([x_test[0]], n_neighbors=2)
    print("预测标签：", y_predict)
    print("最近的 k 个邻居的索引：", k_nearest_indices)
    print("最近的 k 个邻居的距离：", k_nearest_distances)