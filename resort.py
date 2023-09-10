import os
import csv

# 数据集分类

# 载入数据集
database_path = r'.\fer2013'  # 数据集路径
datasets_path = r'.\dataset'  # 输出路径
csv_file = os.path.join(database_path, 'fer2013.csv')  # 数据集
train_csv = os.path.join(datasets_path, 'train.csv')  # 训练集
val_csv = os.path.join(datasets_path, 'val.csv')  # 验证集
test_csv = os.path.join(datasets_path, 'test.csv')  # 测试集

# 分离训练集、验证集和测试集
with open(csv_file) as f:
    csvr = csv.reader(f)  # 按行读取返回行列表
    header = next(csvr)  # 获取第一行标题
    rows = [row for row in csvr]  # 遍历每行

    # 按最后一列的标签将数据集进行分割   第一列row[:-1]，最后一列row[-1]
    trn = [row[:-1] for row in rows if row[-1] == 'Training']
    csv.writer(open(train_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + trn)
    print("训练集的数量为：", len(trn))

    val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
    csv.writer(open(val_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + val)
    print("验证集的数量为：", len(val))

    tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
    csv.writer(open(test_csv, 'w+'), lineterminator='\n').writerows([header[:-1]] + tst)
    print("测试集的数量为：", len(tst))