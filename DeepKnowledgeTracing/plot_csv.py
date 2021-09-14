from matplotlib import pyplot as plt
import numpy as np
import csv

font_title = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 15,
              }
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }

with open("data/full_extract.csv", 'r') as data:
    i = 0
    csv_reader = csv.reader(data)
    tr_cost_loss = []
    tr_card_loss = []
    for row in csv_reader:
        if i != 0:
            tr_cost_loss.append(float(row[1]))
            tr_card_loss.append(float(row[2]))
        i += 1
    print(tr_card_loss)

# training_cost
fig, ax = plt.subplots()
plt.xlabel('Epoch', font1)
plt.ylabel('Loss', font1)
plt.title('Training Loss by Epochs', font_title)

plt.xlim((0, 21))
plt.xticks(np.arange(1, 21))
plt.ylim((0, 20))
plt.yticks(np.arange(0, 24, 4))
# Data for train_cost
x = np.arange(1, 21)
y = tr_cost_loss
ax.plot(x, y, '.', label="cost estimation", markersize='15', linestyle="-", color='dimgrey', linewidth="2")

y = tr_card_loss
ax.plot(x, y, 's', label="cardinality estimation", markersize='6', linestyle="-", color='rosybrown', linewidth="2")

plt.legend(loc='upper right', prop=font2)

fig.savefig("train_loss.png")
plt.show()
