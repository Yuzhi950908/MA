import tensorflow as tf
from tensorflow.keras import layers,losses
from tensorflow.keras.models import Model

#首先可以用各种方法生成随机的数据， 比如具有高斯分布的数据集，定义训练验证集


#构造出 Autoencoder 模型，并且把Autoencoder训练完成。
class Autoencoder(Model):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.latent_dim = 3

        self.encoder = tf.keras.Sequential([
            layers.Input(100),
            layers.Dense(50, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(self.latent_dim, activation='sigmoid')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(25, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(100, activation='relu')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


myAE = Autoencoder()
myAE.compile(optimizer='adam', loss=losses.MeanAbsoluteError())
myAE.optimizer.learning_rate = 0.001
q = myAE.fit(np.array(Xtrain), np.array(Xtrain), epochs=25, shuffle=True)

#然后把电池数据带入，我直接把中间的latent的输出，和LSTM模型输入连接到一起去。这样子我输入的是被压缩后，并且极大程度保留了电池数据特征。这样会提高LSTM模型性能