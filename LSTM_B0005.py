import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.io
from scipy.io import loadmat

#1. validation 做minmax
#2. 别跳数据。我们的数据太少。如果不行的话。就用整个B0005做 训练，整个B0006 做验证集。
#3. 要打乱168数据。因为我只用了前面1衰减到0.9的数据。但是它没有学过最后0.9到0.8的数据。所以它才会预测出来一条横线
#4. 或者直接用这个timeseries_dataset_from_array.
#shuffle
# #5. 或者直接B0005直接拉进去做训练
















###Look at Data
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0005.mat')
#print(list(mat_data.keys()))
#相当于把 B0005.cycle 里的数据里所有cycles提取出来提出来
RD=mat_data['B0005']
#(([[具体的B0005的数据，一共616列]]))
RD=RD[0][0][0][0]
size=range(RD.shape[0])
#print(RD[1])
#RD=[(charge),(('discharge'),('24'),('time'),((V),(C),(t),(c),(v),(t),array([[1.85648742]]))......]
#遍历整个RD，只找有 discharge的Label， 因为只想把Capacity拿出来先做可视化。

#Capacity_firstcycle=RD[1][3][0][0][-1][0][0]
#print(Capacity_firstcycle)

capacity=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity.append(find_capacity)
#print(capacity)
#size_capacity=len(capacity)
#print(size_capacity)


###Visual Data
x=range(len(capacity))
y=capacity
plt.figure()
plt.plot(x,y,marker='o',color='b')
plt.title('Capacity of B0005')
plt.xlabel("Cycles")
plt.ylabel("Capacity")
plt.grid(True)
#plt.show()

###Datapreprocessing###
split_fraction=0.7
train_spilt = int(split_fraction*len(capacity))#117


 #把这函数排除if __name__ = __main__之外
def min_max_normalization(Data):
    min_val = min(Data)
    max_val = max(Data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in Data]
    return normalized_data
train_data=min_max_normalization(capacity[:train_spilt])


###Windowparameter###
past=5
future=4
windowsize=past+future
sequence_length=past


###Training dataset###
num_features=1
start=past+future
end=train_spilt
train_data=train_data[start:end]
batch_size=len(train_data)-windowsize

x_train=[]
for i in range(0,batch_size):
    input=train_data[i:i+past]
    x_train.append(input)
print(x_train)

y_train=[]
for i in range(0,batch_size):
    label=train_data[i+past:i+windowsize]
    y_train.append(label)
print(y_train)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#最重要的步骤，把数据整形成keras需要的样子
x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_train_reshaped = tf.reshape(x_train_tensor, [batch_size, past, num_features])
y_train_reshaped = tf.reshape(y_train_tensor, [batch_size,future, num_features])
print(x_train_reshaped.shape)
print(y_train_reshaped.shape)
#这里x_train_tensor:是99行，然后每行里面有个5个数，shape(99,5)，reshape 成(99,5,1)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#！！！我犯了一个重要的错误，我之前已经自己手动把数据整理成网络要的样子了。没必要这里在整理一遍了。这里在整理一遍，就变成4维的了！！！模型不接受的
dataset_train=keras.preprocessing.timeseries_dataset_from_array(
    x_train_reshaped,
    y_train_reshaped,
    sequence_length=past,#输入5个值
    sampling_rate=1,
    batch_size=batch_size
)

###Validation dataset###
val_split_rate=0.2
val_split=int(len(capacity)*val_split_rate)
start=train_spilt+windowsize       #!没必要。这里keras 把第一个windows框 也避开了，虽然不知道为什么。
end=train_spilt+val_split
Validation_data=capacity[start:end]
#print(len(Validation_data))#24     #一共就24个数据
var_batch_size=len(Validation_data)-windowsize#但是批次是15个就因为碰到最后一个数据就停了
x_val=[]
for i in range(0,var_batch_size):
    input_val=Validation_data[i:i+past]
    x_val.append(input_val)

y_val=[]
for i in range(0,var_batch_size):
    label_val=Validation_data[i+past:i+windowsize]
    y_val.append(label_val)

x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)
x_val_reshaped = tf.reshape(x_val_tensor, [var_batch_size, past, num_features])
y_val_reshaped = tf.reshape(y_val_tensor, [var_batch_size, future, num_features])
print(x_val_reshaped.shape)
print(y_val_reshaped.shape)


#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#！！！我犯了一个重要的错误，我之前已经自己手动把数据整理成网络要的样子了。没必要这里在整理一遍了。这里在整理一遍，就变成4维的了！！！模型不接受的
dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val_reshaped,
    y_val_reshaped,
    sequence_length=past,
    sampling_rate=1,
    batch_size=var_batch_size,
)
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



###LSTM model training###
################################################################################################################################
#input一定是这样的,5个神经元，每个神经元能放N个特征进去。 这里单纯就是塑框架形状，还没有训练.
learning_rate = 0.001

multi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(past, num_features)),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(future * num_features, kernel_initializer=tf.keras.initializers.zeros()),
    tf.keras.layers.Reshape([future, num_features])
])

multi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())
multi_lstm_model.summary()
#我到这里model出来的结果是和keras上的例子一模一样的。
################################################################################################################################


#training
################################################################################################################################
epochs = 50
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

#就这行才是训练核心步骤
history = multi_lstm_model.fit(x_train_reshaped, y_train_reshaped, epochs=epochs,
                               validation_data=(x_val_reshaped,y_val_reshaped), callbacks=[es_callback, modelckpt_callback])

################################################################################################################################


#重点每个epoch 训练集，和验证集是交替训练的。就是为了让模型多见点不同的数据。但是我没想明白，为什么不把验证集也做个标准化呢？除了验证集，连测试集都做标准化要，所有history里就两个keys，一个train_loss, 一个val_loss.
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
#不用分了剩下的全拿去test。
Prediction_data=capacity[train_spilt+val_split:]
Prediction_data=min_max_normalization(Prediction_data)
data_len=len(Prediction_data)
#print(data_len)


batch_size=data_len-windowsize
x_pre=[]
for i in range(0,batch_size):
    input=Prediction_data[i:i+past]
    x_pre.append(input)

#然后把x_pre拿去还是整成keras想要的样子
num_features=1
x_pre_tensor=tf.convert_to_tensor(x_pre, dtype=tf.float32)

x_pre_reshaped=tf.reshape(x_pre_tensor,[batch_size, past, num_features])
#print(x_pre_reshaped.shape),(9, 5, 1)

#获取9组预测值。并且可视化出来和真实值对比
#!!!!!这里我又犯了一个错误预测是一下子9个批次一起进去的，一次性出来一个(9,4,1)的张量。不能一次一次预测的。画图调用即可
predictions = multi_lstm_model.predict(x_pre_reshaped)
#print(predictions)
#调最后一个window, 画图看一下就行了
true_values = Prediction_data[9:9+windowsize]
#print(true_values)
predicted_values = predictions[8]
#print(predicted_values)

# 绘制对比图
plt.scatter(range(1,len(true_values)+1), true_values, label='True Capacity')
plt.scatter([6,7,8,9], predicted_values,label='Predicted Capacity')
plt.title('True Capacity vs Predicted Capacity')
plt.xlabel('Predicted Cycles')
plt.ylabel('Capacity')
plt.legend()
plt.show()


