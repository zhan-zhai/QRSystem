
import torch
from model import DKTModel
import argparse
from data import load_data
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score
import logging
from tqdm import tqdm
import time
from train import train_or_evaluate

logging.basicConfig(level=logging.INFO,
                    filename='DeepKnowledgeTracing/logs/new12.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description='Deep Knowledge tracing model')
parser.add_argument('-train_data_path', type=str, default='data/2015_builder_train.csv', help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str, default='data/2015_builder_test.csv', help='Path to the testing dataset')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('-num_layers', type=int, default=1, help='')
parser.add_argument('-hidden_size', type=int, default=200, help='The number of hidden nodes')
parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('-num_epochs', type=int, default=150, help='The number of epoch')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Clip gradients to this norm')
parser.add_argument('-evaluation_interval', type=int, default=5, help='Evaluation and print result every x epochs')
parser.add_argument("-lw1", "--lambda_w1", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l1-norm.")
parser.add_argument("-lw2", "--lambda_w2", type=float, default=0.00,
                    help="The lambda coefficient for the regularization waviness with l2-norm.")
parser.add_argument("-lr", "--lambda_r", type=float, default=0.00,
                    help="The lambda coefficient for the regularization objective.")
parser.add_argument('-epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer')
args = parser.parse_args()
print(args)


def main():
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
    num_steps = train_max_num_problems
    num_skills = train_max_skill_num
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    print(num_steps, num_skills)
    test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)

    model = DKTModel(num_skills * 2, args.hidden_size, num_skills, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon)

    # 对学习率进行调整
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in tqdm(range(args.num_epochs)):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('-' * 10)
        logging.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logging.info('-' * 10)
        rmse, auc, r2 = train_or_evaluate(model, optimizer, train_students, batch_size, num_steps, num_skills)
        print(rmse, auc, r2)
        logging.info(str(rmse) + " " + str(auc) + " " + str(r2))
        # Testing
        if (epoch + 1) % args.evaluation_interval == 0:
            rmse, auc, r2 = train_or_evaluate(model, optimizer, test_students, batch_size, num_steps, num_skills, False)
            print('Testing')
            logging.info('Testing')
            print(rmse, auc, r2)
            logging.info(str(rmse) + " " + str(auc) + " " + str(r2))

    torch.save(model, 'models/model12.pth')
    torch.save(model.state_dict(), 'params/params12.pth')


if __name__ == "__main__":
    main()
