import csv
import random


def load_data(file_name):
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # row可能是学生做题序列长度或做题序列或答对01序列
            rows.append(row)

    tuple_rows = []
    for index in range(0, len(rows), 3):
        problems_num = int(rows[index][0])  # 题目序列的长度
        tmp_max_skill = max(map(int, rows[index+1]))  # 最大的题目号
        if tmp_max_skill > max_skill_num:
            max_skill_num = tmp_max_skill
        if problems_num <= 2:  # 太少的可以忽略
            continue
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index], rows[index+1], rows[index+2])
            # tup:[题目个数, 题目序列, 答对情况]
            tuple_rows.append(tup)
    # 打乱数据
    random.shuffle(tuple_rows)
    # tuple_rows的每一行是tup:[[题目个数], [题目序列], [答对情况]]
    # max_num_problems最长题目序列
    # max_skill_num是知识点(题目)个数
    return tuple_rows, max_num_problems, max_skill_num + 1


# 将数据转化为one-hot形式
def to_one_hot():
    pass
