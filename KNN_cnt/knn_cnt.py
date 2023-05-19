import csv

from sklearn.neighbors import KNeighborsRegressor

# 打开CSV文件并读取数据
file_path = '../res/future.csv'
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
get_data(file_path)
