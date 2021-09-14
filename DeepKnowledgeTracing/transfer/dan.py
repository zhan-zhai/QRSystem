import sys
sys.path.append('/root/GraduationDesign/DeepKnowledgeTracing')
import torch
from model import DKTModel
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train import train_or_evaluate
from train import mmd_train
from data import load_data
from tqdm import tqdm
import logging
from scipy.stats import ks_2samp

train_data_path = '../data/b_train.csv'
test_data_path = '../data/b_test.csv'
src_data_path = '../data/0910_a_train.csv'

logging.basicConfig(level=logging.INFO,
                    filename='logs/new11.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)
src_students, src_max_num_problems, src_max_skill_num = load_data(src_data_path)

# print(train_students[0], test_students[0])
# print(ks_2samp(train_students[0],test_students[0]).pvalue)

num_steps = train_max_num_problems
num_skills = train_max_skill_num
batch_size = 4
epochs = 150


model = DKTModel(248, 200, 124, 1)
model.load_state_dict(torch.load('../params/params5.pth'))

print("start")


# for param in model.parameters():
#     param.requires_grad = False
#
# model.decoder.requires_grad_()

# criterion = nn.CrossEntropyLoss()
# 仅仅对最后一层进行优化
optimizer = optim.SGD(model.decoder.parameters(), lr=0.01, momentum=0.9)

lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# rmse, auc, r2 = train_or_evaluate(model, optimizer, test_students, batch_size, num_steps, 124, False)
# print('Testing')
# logging.info('Testing')
# print(rmse, auc, r2)
# logging.info(str(rmse) + " " + str(auc) + " " + str(r2))
for epoch in tqdm(range(epochs)):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    # print('-' * 10)
    logging.info('Epoch {}/{}'.format(epoch + 1, epochs))
    logging.info('-' * 10)
    rmse, auc, r2, loss = mmd_train(model, optimizer, train_students, src_students, batch_size, num_steps, 124, lr_scheduler, mmd=True)
    print(rmse, auc, r2, loss)
    logging.info(str(rmse) + " " + str(auc) + " " + str(r2)+" "+str(loss))
    # Testing
    if (epoch + 1) % 5 == 0:
        rmse, auc, r2, loss = mmd_train(model, optimizer, test_students, src_students, batch_size, num_steps, 124, train=False, mmd=True)
        print('Testing')
        logging.info('Testing')
        print(rmse, auc, r2, loss)
        logging.info(str(rmse) + " " + str(auc) + " " + str(r2) + " " + str(loss))


torch.save(model, 'models/model11.pth')
torch.save(model.state_dict(), 'params/params11.pth')

