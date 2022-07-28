from csv import reader
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter
# from math / round

def get_data(path):
    id = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    with open(path) as file:
        csv_reader = reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            id.append(row[0])
            train_acc.append(round(float(row[1]), 2))
            train_loss.append(round(float(row[2]), 2))
            val_acc.append(round(float(row[3]), 2))
            val_loss.append(round(float(row[4]), 2))
    return [id, train_acc, val_acc, train_loss, val_loss]

def plot(x_data, y_data, titles, x_labels, y_labels):
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.plot(x_data, y_data[0])
    plt.title(titles[0])
    plt.xlabel(x_labels[0])
    plt.ylabel(y_labels[0])
    plt.ylim(ymin=0, ymax=1)

    plt.subplot(1, 2, 2) # index 2
    plt.plot(x_data, y_data[1])
    plt.title(titles[1])
    plt.xlabel(x_labels[0])
    plt.ylabel(y_labels[0])
    plt.ylim(ymin=0, ymax=1)
    plt.show()

def draw_plot(path):
    [id, train_acc, val_acc, train_loss, val_loss] = get_data(path)
    plot(id, [train_acc, val_acc], ["Precyzja na zbiorze treningowym", "Precyzja na zbiorze walidacyjnym"], ["nr. epoki"], ['precyzja'])


path = "/home/bdroix/bdroix/aesthetics_analysis/model_history_log_adam_32_2.csv"

draw_plot(path)

