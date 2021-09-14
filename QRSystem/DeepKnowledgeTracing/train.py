import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score
import torch
from model import DKTModel
import argparse
from mmd import mmd_rbf_noaccelerate
from mmd import mmd_rbf_accelerate

lambda1 = 0.01


# 用于初始化隐层
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train_or_evaluate(model, optimizer, students, batch_size, num_steps, num_skills, train=True, scheduler=None, mmd=False):
    # 1.初始化
    total_loss = 0
    input_size = num_skills * 2
    index = 0
    actual_labels = []
    pred_labels = []
    hidden = model.init_hidden(num_steps)
    count = 0
    batch_num = len(students) // batch_size
    rmse = 0
    auc = 0
    r2 = 0

    # 2. 获取每个batch训练的数据集
    # x: 批量喂入的数据集
    # target_id: 数据集的题号(喂入一个批量的数据集, 模型输出是所有知识点的掌握情况, 需要计算准确率的是本批次的题号)
    # target_correctness: 对应target_id的对错
    while index + batch_size < len(students):
        # s = time.time()
        x = np.zeros((num_steps, batch_size))
        # target_id: List[int] = []
        target_id = []
        # current_id = []
        # current_correctness = []
        target_correctness = []
        for i in range(batch_size):
            # student: [[题目个数], [题目序列], [答对情况]]
            student = students[index + i]
            problem_ids = student[1]
            correctness = student[2]
            for j in range(len(problem_ids)-1):
                problem_id = int(problem_ids[j])
                correct = int(correctness[j])
                # problem_id * 2 + correct
                x[j, i] = problem_id * 2 + correct

                # if int(correctness[j]) == 0:
                #     label_index = problem_id
                # else:
                #     label_index = problem_id + num_skills
                # x[j, i] = label_index
                # 需要预测的是答题序列的后n-1个(t时刻需要预测t+1时刻) ???
                target_id.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j + 1]))

                # current_id.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j]))
                # current_correctness.append(correct)
                target_correctness.append(int(correctness[j + 1]))
                actual_labels.append(int(correctness[j + 1]))

        index += batch_size
        count += 1
        target_id = torch.tensor(target_id, dtype=torch.int64)
        # current_id = torch.tensor(current_id, dtype=torch.int64)
        target_correctness = torch.tensor(target_correctness, dtype=torch.float)
        # current_correctness = torch.tensor(current_correctness, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.int64)
        x = torch.unsqueeze(x, 2)
        input_data = torch.FloatTensor(num_steps, batch_size, input_size)
        input_data.zero_()
        # scatter_用于生成one_hot向量，这里会将所有答题序列统一为num_steps
        input_data.scatter_(2, x, 1)

        if train:
            # 初始化隐层，相当于hidden = m.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)
            # 把模型中参数的梯度设为0
            optimizer.zero_grad()
            # 前向计算, output:(num_steps, batch_size, num_skills)
            output, hidden = model(input_data, hidden)
            # 将输出层转化为一维张量
            output = output.contiguous().view(-1)
            # print(output.size())
            # tf.gather用一个一维的索引数组，将张量中对应索引的向量提取出来
            logits = torch.gather(output, 0, target_id)

            # 预测, preds是0~1的数组
            preds = torch.sigmoid(logits)

            for p in preds:
                pred_labels.append(p.item())

            # 计算误差，相当于nn.functional.binary_cross_entropy_with_logits()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)

            # logits_r = torch.gather(output, 0, current_id)
            # #  r
            # loss_r = criterion(logits_r, current_correctness)
            # loss += loss_r * 0.2
            #
            # #  l1-norm
            # output = torch.reshape(output, (num_steps, num_skills, -1))
            # waviness_norm_l1 = torch.abs(output[1:, :, :] - output[:-1, :, :])
            # waviness_l1 = torch.sum(waviness_norm_l1) / num_steps / num_skills
            # loss += waviness_l1 * 1.0
            #
            # #  l2-norm
            # waviness_norm_l2 = torch.square(output[1:, :, :] - output[:-1, :, :])
            # waviness_l2 = torch.sum(waviness_norm_l2) / num_steps / num_skills
            # loss += waviness_l2 * 30.0

            # 反向传播
            if mmd:
                loss += lambda1 * mmd_rbf_noaccelerate(logits, target_correctness)
            loss = loss.requires_grad_()
            loss.backward()
            # 梯度截断，防止在RNNs或者LSTMs中梯度爆炸的问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()

            total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

        else:
            with torch.no_grad():
                # 前向计算
                model.eval()
                output, hidden = model(input_data, hidden)
                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)
                preds = torch.sigmoid(logits)
                for p in preds:
                    pred_labels.append(p.item())

                # 计算误差，但不进行反向传播
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()
                hidden = repackage_hidden(hidden)
        # 打印误差等信息

        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print("Batch {}/{} AUC: {}".format(count, batch_num, auc))
        # logging.info("Batch {}/{} AUC: {}".format(count, batch_num, auc))

        r2 = r2_score(actual_labels, pred_labels)

        # e = time.time()
        # print('时间', e-s)



    return rmse, auc, r2


def mmd_train(model, optimizer, students, src_students, batch_size, num_steps, num_skills, train=True, scheduler=None, mmd=False):
    # 1.初始化
    total_loss = 0
    input_size = num_skills * 2
    index = 0
    actual_labels = []
    pred_labels = []
    hidden = model.init_hidden(num_steps)
    count = 0
    batch_num = len(students) // batch_size
    rmse = 0
    auc = 0
    r2 = 0

    # 2. 获取每个batch训练的数据集
    # x: 批量喂入的数据集
    # target_id: 数据集的题号(喂入一个批量的数据集, 模型输出是所有知识点的掌握情况, 需要计算准确率的是本批次的题号)
    # target_correctness: 对应target_id的对错
    while index + batch_size < len(students):
        # s = time.time()
        x = np.zeros((num_steps, batch_size))
        src_x = np.zeros((num_steps, batch_size))
        # target_id: List[int] = []
        target_id = []
        # current_id = []
        # current_correctness = []
        target_correctness = []
        for i in range(batch_size):
            # student: [[题目个数], [题目序列], [答对情况]]
            student = students[index + i]
            src_student = src_students[index + i]
            problem_ids = student[1]
            src_problem_ids = src_student[1]
            correctness = student[2]
            src_correctness = src_student[2]
            for j in range(len(problem_ids) - 1):
                problem_id = int(problem_ids[j])
                src_problem_id = int(problem_ids[j])
                correct = int(correctness[j])
                src_correct = int(correctness[j])
                # problem_id * 2 + correct
                x[j, i] = problem_id * 2 + correct
                src_x[j, i] = src_problem_id * 2 + src_correct

                # if int(correctness[j]) == 0:
                #     label_index = problem_id
                # else:
                #     label_index = problem_id + num_skills
                # x[j, i] = label_index
                # 需要预测的是答题序列的后n-1个(t时刻需要预测t+1时刻) ???
                target_id.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j + 1]))

                # current_id.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j]))
                # current_correctness.append(correct)
                target_correctness.append(int(correctness[j + 1]))
                actual_labels.append(int(correctness[j + 1]))

        index += batch_size
        count += 1
        target_id = torch.tensor(target_id, dtype=torch.int64)
        # current_id = torch.tensor(current_id, dtype=torch.int64)
        target_correctness = torch.tensor(target_correctness, dtype=torch.float)
        # current_correctness = torch.tensor(current_correctness, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.int64)
        src_x = torch.tensor(src_x, dtype=torch.int64)
        x = torch.unsqueeze(x, 2)
        src_x = torch.unsqueeze(src_x, 2)
        input_data = torch.FloatTensor(num_steps, batch_size, input_size)
        src_input_data = torch.FloatTensor(num_steps, batch_size, input_size)
        input_data.zero_()
        src_input_data.zero_()
        # scatter_用于生成one_hot向量，这里会将所有答题序列统一为num_steps
        input_data.scatter_(2, x, 1)
        src_input_data.scatter_(2, src_x, 1)

        if train:
            # 初始化隐层，相当于hidden = m.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)
            # 把模型中参数的梯度设为0
            optimizer.zero_grad()
            # 前向计算, output:(num_steps, batch_size, num_skills)
            output, hidden = model(input_data, hidden)
            src_output, src_hidden = model(src_input_data, hidden)

            gather_src_out = src_output[:10, :10]
            gather_out = output[:10, :10]
            mmd_loss = mmd_rbf_noaccelerate(gather_src_out, gather_out)

            # 将输出层转化为一维张量
            output = output.contiguous().view(-1)
            src_output = src_output.contiguous().view(-1)
            # print(output.size())
            # tf.gather用一个一维的索引数组，将张量中对应索引的向量提取出来
            logits = torch.gather(output, 0, target_id)

            # 预测, preds是0~1的数组
            preds = torch.sigmoid(logits)

            for p in preds:
                pred_labels.append(p.item())

            # 计算误差，相当于nn.functional.binary_cross_entropy_with_logits()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)

            # logits_r = torch.gather(output, 0, current_id)
            # #  r
            # loss_r = criterion(logits_r, current_correctness)
            # loss += loss_r * 0.2
            #
            # #  l1-norm
            # output = torch.reshape(output, (num_steps, num_skills, -1))
            # waviness_norm_l1 = torch.abs(output[1:, :, :] - output[:-1, :, :])
            # waviness_l1 = torch.sum(waviness_norm_l1) / num_steps / num_skills
            # loss += waviness_l1 * 1.0
            #
            # #  l2-norm
            # waviness_norm_l2 = torch.square(output[1:, :, :] - output[:-1, :, :])
            # waviness_l2 = torch.sum(waviness_norm_l2) / num_steps / num_skills
            # loss += waviness_l2 * 30.0

            # 反向传播
            if mmd:
                loss += lambda1 * mmd_loss
            loss = loss.requires_grad_()
            loss.backward()
            # 梯度截断，防止在RNNs或者LSTMs中梯度爆炸的问题
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()

            total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

        else:
            with torch.no_grad():
                # 前向计算
                model.eval()
                output, hidden = model(input_data, hidden)
                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)
                preds = torch.sigmoid(logits)
                for p in preds:
                    pred_labels.append(p.item())

                # 计算误差，但不进行反向传播
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()
                hidden = repackage_hidden(hidden)
        # 打印误差等信息

        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print("Batch {}/{} AUC: {}".format(count, batch_num, auc))
        # logging.info("Batch {}/{} AUC: {}".format(count, batch_num, auc))

        r2 = r2_score(actual_labels, pred_labels)

        # e = time.time()
        # print('时间', e-s)

    return rmse, auc, r2


# skill_num: 题号
# return 该题目答对的概率
def predict(model, skill_num, records, num_steps, num_skills, lstm_size):
    hidden = model.init_hidden(num_steps)
    input_size = lstm_size * 2
    x = np.zeros((num_steps, 1))
    record = records[0]
    problems_ids = record[1]
    correctness = record[2]
    for i in range(len(problems_ids)):
        problems_id = int(problems_ids[i])
        correct = int(correctness[i])
        x[i, 0] = problems_id * 2 + correct
    x = torch.tensor(x, dtype=torch.int64)
    x = torch.unsqueeze(x, 2)
    input_data = torch.FloatTensor(num_steps, 1, input_size)
    input_data.zero_()
    # scatter_用于生成one_hot向量，这里会将所有答题序列统一为num_steps
    input_data.scatter_(2, x, 1)
    p = 0
    average_p = 0
    with torch.no_grad():
        # 前向计算
        model.eval()
        output, hidden = model(input_data, hidden)
        output = torch.sigmoid(output)
        p = output[num_steps-1][skill_num-1]
        total_p = 0
        # for i in range(num_skills):
        #     total_p += output[num_steps-1][i]
        # average_p = total_p / num_skills

        average_p = torch.mean(output)

    return p, average_p
