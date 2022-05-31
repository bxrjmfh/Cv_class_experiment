import datetime
import os
import numpy
import tensorflow as tf
import tensorflow_addons as tfa
from expriment_4.packet.load_data import *
from tensorflow.keras.utils import to_categorical


def Model_784_300_100_10(name='MNIST', learn_rate_1=1e-2,
                         learn_rate_2=1e-2, learn_rate_3=1e-2, epochs=20, batch_size=6400):
    dir_name = name + '_' + str(learn_rate_1) + '_' + str(learn_rate_2) + \
               '_' + str(learn_rate_3) + "_ep_" + str(epochs)
    try:
        os.mkdir(dir_name)
    except:
        print('exist!')

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
        tf.keras.optimizers.Adam(learning_rate=learn_rate_2),
        tf.keras.optimizers.Adam(learning_rate=learn_rate_3)
    ]
    # 设置优化器参数
    optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])
        , (optimizers[2], model.layers[2])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels))

    import matplotlib.pyplot as plt
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


def Model_784_300_10(name='MNIST', learn_rate_1=1e-2,
                     learn_rate_2=1e-2, epochs=20, batch_size=6400):
    dir_name = name + '_' + str(learn_rate_1) + '_' + str(learn_rate_2) + \
               '_twolayer_ep_' + str(epochs)
    try:
        os.mkdir(dir_name)
    except:
        print('exist!')

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(784,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
        tf.keras.optimizers.Adam(learning_rate=learn_rate_2),
    ]
    # 设置优化器参数
    optimizers_and_layers = [(optimizers[0], model.layers[0])
                            , (optimizers[1], model.layers[1])
                             ]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(train_images, train_labels,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels))

    import matplotlib.pyplot as plt
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


def Get_10_results_300_100_10():
    epochs = 20
    batch_size = 6400
    learn_rate_1 = 1e-2
    learn_rate_2 = 1e-2
    learn_rate_3 = 1e-3
    acc_result = []
    val_acc_result = []
    for i in range(10):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
            tf.keras.optimizers.Adam(learning_rate=learn_rate_2),
            tf.keras.optimizers.Adam(learning_rate=learn_rate_3)
        ]
        # 设置优化器参数
        optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])
            , (optimizers[2], model.layers[2])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        history = model.fit(train_images, train_labels,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(test_images, test_labels))
        acc_result.append(history.history['accuracy'][-1])
        val_acc_result.append(history.history['val_accuracy'][-1])
    return acc_result,val_acc_result


def Get_10_results_300_10():
    epochs = 20
    batch_size = 6400
    learn_rate_1 = 1e-3
    learn_rate_2 = 1e-3
    acc_result = []
    val_acc_result = []
    for i in range(10):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ])
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=learn_rate_1),
            tf.keras.optimizers.Adam(learning_rate=learn_rate_2),
        ]
        # 设置优化器参数
        optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1])
            ]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        history = model.fit(train_images, train_labels,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(test_images, test_labels))
        acc_result.append(history.history['accuracy'][-1])
        val_acc_result.append(history.history['val_accuracy'][-1])
    return acc_result, val_acc_result

# 加载MNIST数据集
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 784))
train_images = train_images.astype("float32") / 255
# 归一化，便于训练
test_images = test_images.reshape((10000, 784))
test_images = test_images.astype("float32") / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 转化成独热编码

# 测试学习率
# Model_784_300_100_10(learn_rate_1=0.1,learn_rate_2=0.1,learn_rate_3=0.1)
# Model_784_300_10(learn_rate_1=0.1,learn_rate_2=0.01)
# Model_784_300_10(learn_rate_1=0.01,learn_rate_2=0.01)
# Model_784_300_10(learn_rate_1=0.1,learn_rate_2=0.1)
# Model_784_300_10(learn_rate_1=0.001,learn_rate_2=0.1)
# Model_784_300_10(learn_rate_1=0.001,learn_rate_2=0.001)

# 输出每次的训练结果
accs_300_100_10,val_accs_300_100_10 = Get_10_results_300_100_10()
acc_300_10,val_accs_300_10 = Get_10_results_300_10()