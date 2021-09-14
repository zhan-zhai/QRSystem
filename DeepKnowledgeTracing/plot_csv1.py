from matplotlib import pyplot as plt
import numpy as np
import csv

font_title = {'family': 'sans-serif',
              'weight': 'normal',
              'size': 15,
              }
font1 = {'family': 'sans-serif',
         'weight': 'normal',
         'size': 13,
         }
font2 = {'family': 'sans-serif',
         'weight': 'normal',
         'size': 13,
         }

with open("result/mmd_loss.csv", 'r') as data:
    csv_reader = csv.reader(data)
    loss = []
    for row in csv_reader:
        loss.append(float(row[0]))
    # print(tr_card_loss)

# training_cost
fig, ax = plt.subplots()
plt.xlabel('批次', font1)
plt.ylabel('MMD距离', font1)
plt.title('第一次迭代训练的MMD距离', font_title)

plt.xlim((0, 9))
plt.xticks(np.arange(1, 9, 1))
# plt.ylim((0.4, 0.7))
plt.yticks(np.arange(0.1, 0.3, 0.02))
# Data for train_cost
x = np.arange(1, 9, 1)
y = loss
ax.plot(x, y, '.', label="期望最大推荐算法", markersize='15', linestyle="-", color='dimgrey', linewidth="2")

# y = tr_cost_loss_2
# ax.plot(x, y, 's', label="随机推荐算法", markersize='10', linestyle="-", color='dimgrey', linewidth="2")
#
# y = tr_cost_loss_3
# ax.plot(x, y, 'v', label="顺序推荐算法", markersize='13', linestyle="-", color='dimgrey', linewidth="2")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus']=False    # 这两行需要手动设置
# plt.legend(loc='upper left', prop=font2)

fig.savefig("photo/mmd_loss-cn.png")
plt.show()
