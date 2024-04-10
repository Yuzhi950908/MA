import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.io
from scipy.io import loadmat

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
#print(x_train)

y_train=[]
for i in range(0,batch_size):
    label=train_data[i+past:i+windowsize]
    y_train.append(label)
#print(y_train)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#最重要的步骤，把数据整形成keras需要的样子
x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_train_reshaped = tf.reshape(x_train_tensor, [batch_size, past, num_features])
y_train_reshaped = tf.reshape(y_train_tensor, [batch_size, future, num_features])
#print(x_train_reshaped.shape)
#这里x_train_tensor:是99行，然后每行里面有个5个数
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

dataset_train=keras.preprocessing.timeseries_dataset_from_array(
    x_train_reshaped,
    y_train_reshaped,
    sequence_length=past,#输入5个值
    sampling_rate=1,
    batch_size=batch_size
)

###Validation dataset###
var_split_rate=0.2
var_split=int(len(capacity)*var_split_rate)
start=train_spilt+windowsize       #!这里keras 把第一个windows框 也避开了，虽然不知道为什么。
end=train_spilt+var_split
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

x_val_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_val_reshaped = tf.reshape(x_train_tensor, [batch_size, past, num_features])
y_val_reshaped = tf.reshape(y_train_tensor, [batch_size, future, num_features])

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val_reshaped,
    y_val_reshaped,
    sequence_length=sequence_length,
    sampling_rate=1,
    batch_size=var_batch_size,
)


###LSTM model training###
################################################################################################################################
#input一定是这样的,5个神经元，每个神经元能放N个特征进去。 这里单纯就是塑框架形状，还没有训练.
learning_rate = 0.001
#？？？？这里 input layer期待的是数据是(99批，一批5步，至于none意味着每步里有多少特征无所谓
inputs = keras.layers.Input(shape=(x_train_reshaped.shape[0], x_train_reshaped.shape[1]))
#中间暂时设定32个LSTM unit，之后可以改
lstm_out = keras.layers.LSTM(32)(inputs)
#一次预测4个步
outputs = keras.layers.Dense(4)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()
################################################################################################################################




#training
################################################################################################################################

epochs = 50

#为了防止overfit，只要权重不咋变了，就结束，不要在训练了。所以设一个 call_funktion
#相当于保存weights的快照的空文件，之后训练完之后，可以权重可视化。
path_checkpoint = "lstm_model_checkpoint.weights.h5"
#？？？？？？预先设定好参数,不允许它小于0，然后如果2个epoch，权重都不变了。就结束直接。但是我不知道为什么epoch应该48次就该停了，但它没停下来，也没报错
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

#就这几行才是训练核心步骤
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)
print(history)
################################################################################################################################


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