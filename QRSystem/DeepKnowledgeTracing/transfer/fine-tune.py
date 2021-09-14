import sys
sys.path.append('/root/GraduationDesign/DeepKnowledgeTracing')
import torch
from model import DKTModel
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train import train_or_evaluate
from data import load_data
from tqdm import tqdm
import logging

train_data_path = '../data/CAT_train.csv'
test_data_path = '../data/CAT_test.csv'

logging.basicConfig(level=logging.INFO,
                    filename='logs/new3.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)

num_steps = train_max_num_problems
num_skills = train_max_skill_num
batch_size = 32
epochs = 150


model = DKTModel(248, 200, 124, 1)
model.load_state_dict(torch.load('../params/params5.pth'))

print("start")


for param in model.parameters():
    param.requires_grad = False

model.decoder.requires_grad_()

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
    rmse, auc, r2 = train_or_evaluate(model, optimizer, train_students, batch_size, num_steps, 124, lr_scheduler)
    print(rmse, auc, r2)
    logging.info(str(rmse) + " " + str(auc) + " " + str(r2))
    # Testing
    if (epoch + 1) % 5 == 0:
        rmse, auc, r2 = train_or_evaluate(model, optimizer, test_students, batch_size, num_steps, 124, train=False)
        print('Testing')
        logging.info('Testing')
        print(rmse, auc, r2)
        logging.info(str(rmse) + " " + str(auc) + " " + str(r2))

torch.save(model, 'models/model3.pth')
torch.save(model.state_dict(), 'params/params3.pth')

