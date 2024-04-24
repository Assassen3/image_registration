import tensorflow as tf
from tensorflow.keras.layers import Dense


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.D1 = Dense(10, activation='relu')
        self.D2 = Dense(10, activation='relu')
        self.D3 = Dense(1, activation='sigmoid')
        self.D4 = Dense(1, activation='sigmoid')

    def call(self, inputs, **kwargs):
        o1 = self.D3(self.D1(inputs))
        o2 = self.D4(self.D2(inputs))
        return o1, o2


# 定义一个简单的模型
model = Net()


# 函数定义一个自定义的metric，仅接受y_pred
def mean_pred(y_pred, _):
    print(_.shape)
    return tf.reduce_mean(y_pred)


# 编译模型，并添加不同类型的metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[[], [mean_pred]])

# 模拟一些数据
import numpy as np

x_train = np.random.random((100, 20))
y_train = (np.random.randint(2, size=(100, 1)), np.random.randint(2, size=(100, 1)))

y = model(x_train)
# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=50)
