import datetime
import os
import numpy
import tensorflow as tf
import tensorflow_addons as tfa
from expriment_4.packet.load_data import *
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def Draw_each_time(acc, val_acc, name):
    numbers = range(1, len(acc) + 1)
    ave_acc = sum(acc) / len(acc)
    ave_val_acc = sum(val_acc) / len(val_acc)
    plt.title(name + "_acc")
    plt.plot(numbers, acc, 'bo', label="train_acc")
    plt.plot(numbers, val_acc, 'go', label="validation_acc")
    plt.axhline(y=ave_acc, c='b', label="ave_train_acc_{:.3f}".format(ave_acc))
    plt.axhline(y=ave_val_acc, c='g', label="ave_val_acc_{:.3f}".format(ave_val_acc))
    plt.xlabel("numbers")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig(name)
    plt.show()

def Get_10_results_4_10_3():
    epochs = 200
    learn_rate_1 = 1e-4
    learn_rate_2 = 1e-2
    batch_size = 32
    acc_result = []
    val_acc_result = []
    for i in range(10):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax'),
        ])
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
            tf.keras.optimizers.Adam(learning_rate=learn_rate_2)
        ]
        optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(training_set, training_label,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(test_set, test_label))

        acc_result.append(history.history['accuracy'][-1])
        val_acc_result.append(history.history['val_accuracy'][-1])
    return acc_result, val_acc_result

def Model_4_10_3(name='iris', learn_rate_1=1e-4,
                 learn_rate_2=1e-2, epochs=100, batch_size=32):
    dir_name = name + '_' + str(learn_rate_1) + '_' + str(learn_rate_2) + "_ep_" + str(epochs)
    try:
        os.mkdir(dir_name)
    except:
        print('exist!')

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
        tf.keras.optimizers.Adam(learning_rate=learn_rate_2)
    ]
    # 设置优化器参数
    optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(training_set, training_label,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(test_set, test_label))

    loss_1 = history.history['loss']
    val_loss_1 = history.history['val_loss']
    epochs = range(1, len(loss_1) + 1)
    plt.plot(epochs, loss_1, 'bo', label="train_loss in 1")
    plt.title('loss_' + dir_name)
    plt.plot(epochs, val_loss_1, "b", label="validation loss in 1")
    plt.xlabel("Epoches")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(dir_name + '/loss_' + datetime.datetime.now().strftime('%H_%M_%S'))
    plt.show()
    # 测试训练损失

    acc_1 = history.history['accuracy']
    val_acc_1 = history.history["val_accuracy"]
    plt.plot(epochs, acc_1, 'bo', label='train acc')
    plt.plot(epochs, val_acc_1, 'b', label='validation acc')
    plt.xlabel("epoches")
    plt.ylabel("acc")
    plt.title('acc_' + dir_name)
    plt.legend()
    plt.savefig(dir_name + '/acc_' + datetime.datetime.now().strftime('%H_%M_%S'))
    plt.show()
    # 测试训练精度


# 加载鸢尾花数据集
dataset, labelset = load_the_csv('iris.csv')
for i in range(len(dataset[0])):
    conv_string_to_float(dataset, i)
# 将string格式转化为浮点型
normalize(dataset)
# 将数据标准化
type_dic = conv_string_map_integer(labelset)
# 将字符串类型的数据标签映射为int类型便于运算
dataset = numpy.asarray(dataset)
training_set = dataset[:75]
training_label = labelset[:75]
# 划分训练集以及对应标签
test_set = dataset[75:]
test_label = labelset[75:]
# 划分测试集以及对应标签
training_label = to_categorical(training_label)
test_label = to_categorical(test_label)
# 编码为独热玛
x_val = training_set[:10]
training_set = training_set[10:]
y_val = training_label[:10]
training_label = training_label[10:]
# 在训练集中划分验证集

# model with some rate

# Model_4_10_3(learn_rate_1=1e-5, learn_rate_2=1e-5, epochs=2000)

acc,val_acc = Get_10_results_4_10_3()
Draw_each_time(acc,val_acc,"4x10x3")
