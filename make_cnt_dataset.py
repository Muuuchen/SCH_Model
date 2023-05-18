import csv

f = open('res/future.csv',encoding='utf-8')
csv_writer = csv.writer(f)

csv_writer.writerow(["姓名","年龄","性别"])

# 4. 写入csv文件内容
csv_writer.writerow(["l",'18','男'])
csv_writer.writerow(["c",'20','男'])
csv_writer.writerow(["w",'22','女'])

# 5. 关闭文件
f.close()

