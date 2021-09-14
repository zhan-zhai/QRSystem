from data import load_data
from model import DKTModel
from train import predict
import torch
import copy
import random


def qr(qr_students):
    train_data_path = './DeepKnowledgeTracing/data/b.csv'
    qr_data_path = './DeepKnowledgeTracing/data/QR_test.csv'
    students, max_num_problems, max_skill_num = load_data(train_data_path)
    qa_records = qr_students
    num_problems = len(qa_records[0][1])
    model = DKTModel(248, 250, 124, 1)
    model.load_state_dict(torch.load('./DeepKnowledgeTracing/transfer/params/params2.pth'))

    # 平均概率计算也就是预测从1到max_skill_num的预测值的平均
    true_answers = []
    for i in range(num_problems):
        if qa_records[0][2][i] == '1':
            true_answers.append(qa_records[0][1][i])
    print(true_answers)
    # for j in range(10):
    #     # 每次的题目推荐
    #     tr = 0
    #     target = 0
    #     target_p = 0.0
    #     # for i in range(1, max_skill_num+1):
    #     #     p, a_p = predict(model, i, qa_records, num_problems, max_skill_num, 124)
    #     #     if str(i) in true_answers:
    #     #         continue
    #     #     true_records = copy.deepcopy(qa_records)
    #     #     true_records[0][1].append(str(i))
    #     #     true_records[0][2].append('1')
    #     #     false_records = copy.deepcopy(qa_records)
    #     #     false_records[0][1].append(str(i))
    #     #     false_records[0][2].append('0')
    #     #     p1, average_p_true = predict(model, i, true_records, num_problems+1, max_skill_num, 124)
    #     #     p2, average_p_false = predict(model, i, false_records, num_problems+1, max_skill_num, 124)
    #     #     er = p * average_p_true + (1-p) * average_p_false
    #     #     if er > tr:
    #     #         target = i
    #     #         tr = er
    #     #         target_p = p
    #     end = int(true_answers[len(true_answers)-1])
    #     if end >= 93:
    #         target = 1
    #     else:
    #         target = end+1
    #     target_p, a_p = predict(model, target, qa_records, num_problems, max_skill_num, 124)
    #     a = random.random()
    #     qa_records[0][1].append(str(target))
    #     if a <= target_p:
    #         qa_records[0][2].append('1')
    #         true_answers.append(str(target))
    #         # print(true_answers)
    #     else:
    #         qa_records[0][2].append('0')
    #     num_problems += 1
    #     final_p, final_average_p = predict(model, target, qa_records, num_problems, max_skill_num, 124)
    #     print('第'+str(j+1)+'推荐，推荐题目为'+str(target))
    #     print('平均概率为', final_average_p, final_p)
    #     true_answers.append(str(target))

    # 每次的题目推荐
    tr = 0
    target = 0
    for i in range(1, max_skill_num + 1):
        p,  a_p= predict(model, i, qa_records, num_problems, max_skill_num, 124)
        if str(i) in true_answers:
            continue
        true_records = copy.deepcopy(qa_records)
        true_records[0][1].append(str(i))
        true_records[0][2].append('1')
        false_records = copy.deepcopy(qa_records)
        false_records[0][1].append(str(i))
        false_records[0][2].append('0')
        p1, average_p_true = predict(model, i - 1, true_records, num_problems + 1, max_skill_num, 124)
        p2, average_p_false = predict(model, i - 1, false_records, num_problems + 1, max_skill_num, 124)
        er = p * average_p_true + (1 - p) * average_p_false
        if er > tr:
            target = i
            tr = er
    return target, tr
