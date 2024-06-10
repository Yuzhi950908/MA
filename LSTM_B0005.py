import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.io
from scipy.io import loadmat
from DWT_All_Capacity import reconstructed_signal_B0005

#全数据集做normalization
#keras.preprocessing.timeseries_dataset_from_array 滑动窗口对Input Label提取出数据，这里不洗牌
#np.random 整个提取出的INput 和 Label 数据集去洗牌，然后再切割成训练。和Testing的数据集
#fit 训练，然后训练的同时我只要定一个 Validation split rate 它自动帮我做验证
#然后test 去做预测即可

#Input Data
capacity=reconstructed_signal_B0005

###全数据集做normalization
def min_max_normalization(Data):
    min_val = min(Data)
    max_val = max(Data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in Data]
    return normalized_data
capacity=min_max_normalization(capacity[:])


#plt.figure(figsize=(10,6))
#plt.plot(capacity,marker='o',color='b',linestyle='-')
#plt.title('capacity after normalization')
#plt.xlabel('cycles')
#plt.ylabel('capacity')
#plt.show()







###抽取Inputdata
past=5
future=4
windowsize=past+future
batch_size=len(capacity)-windowsize#159

#这里是用滑动窗口的方式，提取出了159个批次的Inputdata数据
Dataset_input = tf.keras.preprocessing.timeseries_dataset_from_array(
    capacity,
    None,# 这里本来应该定义标签的，但是我不需要，我后面要做打乱的
    sequence_length=past,#这里其实本来应该定义标签的长度，但是我这里没有标签，所以出来只有inputdata
    sampling_rate=1,
    batch_size=past,#这里填159没用的，因为这里定义的是每个批次的想要的长度
)
######################################################################
input_data = []
for batch in Dataset_input.as_numpy_iterator():
    input_data.extend(batch)
#手动截成159个数据
input_data = input_data[:batch_size]

#labeldata
label_data=capacity[past:]
Dataset_Label=tf.keras.preprocessing.timeseries_dataset_from_array(
    label_data,
    None,
    sequence_length=future,
    sampling_rate=1,
    batch_size=future
)
label_data=[]
for batch in Dataset_Label.as_numpy_iterator():
    label_data.extend(batch)
#手动截成159个数据
label_data=label_data[:batch_size]

#将batchsize 给他对应起来做乱序
tf.random.set_seed(4)#不变的随机数
shuffled_indices=tf.random.shuffle(tf.range(batch_size))
shuffled_input_data = tf.gather(input_data, shuffled_indices)
shuffled_label_data = tf.gather(label_data, shuffled_indices)

#print(shuffled_indices)



###Split Train_dataset and Test_dataset
train_split=0.9
test_split=0.1
train_input = shuffled_input_data[:int(train_split*batch_size)]
train_label = shuffled_label_data[:int(train_split*batch_size)]
test_input = shuffled_input_data[int(train_split*batch_size):]
test_label = shuffled_label_data[int(train_split*batch_size):]
train_input=tf.expand_dims(train_input, axis=-1)
train_label=tf.expand_dims(train_label, axis=-1)
test_input=tf.expand_dims(test_input, axis=-1)
test_label=tf.expand_dims(test_label, axis=-1)
###LSTM model training###
################################################################################################################################
#input一定是这样的,无所谓输入多少组sample，但每个神经元能放N个特征进去。 这里单纯就是塑框架形状，还没有训练.
learning_rate = 0.001
num_features=1
multi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(past, num_features),activation='tanh', recurrent_activation='sigmoid'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(future * num_features, activation='relu'),
    tf.keras.layers.Reshape([future, num_features])
])


print(multi_lstm_model.input_shape)
#这儿有问题， 不应该是None， none ，1
# 加一个L1 或者 L2正则项 --> 为了避免 有一些权重过大 或者 过小 的情况
# 只有Dense 层有激活函数，LSTM 没有考虑到！这样设置的话。并且不可以relu，选别的， 值域 还不可以是复数。
# 最好的方式 是通过 官网的例子学习，甚至可以去看Matlab



multi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError())
multi_lstm_model.summary()
################################################################################################################################


#training
################################################################################################################################
epochs = 100
#为了防止overfit，只要权重不咋变了，就结束，不要在训练了。所以设一个 call_funktion
#相当于保存weights的快照的空文件，之后训练完之后，可以权重可视化。
path_checkpoint = "lstm_model_checkpoint.weights.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=2)


#从这里开始才是模型检查，之前都是参数预设相当于。
#在 Keras 的 ModelCheckpoint 回调函数中，verbose 参数用于控制日志输出的详细程度。并且保存在path中
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

#就才是训练核心步骤
history = multi_lstm_model.fit(
    train_input,
    train_label,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[es_callback, modelckpt_callback]
)
################################################################################################################################


#重点每个epoch 训练集，和验证集是交替训练的。就是为了让模型多见点不同的数据。所有history里就两个keys，一个train_loss, 一个val_loss.
#就必须要用loss=history.history["loss"]把train loss的不同epoch的值调出来。所以epochs和loss 长度一摸一样的！这个可视化代码不会改变的可以直接记住它。以后需要参考就好了

#weight visualize
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

visualize_loss(history, "Training and Validation Loss")


###Prediction###
#统共16组预测值。输入预测
prediction_data = multi_lstm_model.predict(test_input)

print(prediction_data)
# 分别绘制5幅对比图
for i in range(5):
    #所谓window= test_input+ label
    test_input_window = test_input[i]
    test_label_window = test_label[i]
    true_values=np.concatenate((test_input_window,test_label_window),axis=0)
    #调出预测数据
    predicted_values = prediction_data[i]
    plt.scatter(range(1,len(true_values)+1), true_values, label='True Capacity')
    plt.scatter([6,7,8,9], predicted_values,label='Predicted Capacity')
    plt.title('True Capacity vs Predicted Capacity')
    plt.xlabel('Predicted Cycles')
    plt.ylabel('Capacity')
    plt.legend()
    plt.show()

