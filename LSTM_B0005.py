import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.io
from scipy.io import loadmat
from DWT_All_Capacity import reconstructed_signal

#全数据集做normalization
#keras.preprocessing.timeseries_dataset_from_array 滑动窗口对Input Label提取出数据，这里不洗牌
#np.random 整个提取出的INput 和 Label 数据集去洗牌，然后再切割成训练。和Testing的数据集
#fit 训练，然后训练的同时我只要定一个 Validation split rate 它自动帮我做验证
#然后test 去做预测即可

#Input Data
capacity=reconstructed_signal

###全数据集做normalization
def min_max_normalization(Data):
    min_val = min(Data)
    max_val = max(Data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in Data]
    return normalized_data
capacity=min_max_normalization(capacity[:])

###抽取Inputdata
past=5
future=4
windowsize=past+future
batch_size=len(capacity)-windowsize
Dataset_input = tf.keras.preprocessing.timeseries_dataset_from_array(
    capacity,
    None,
    sequence_length=past,
    sampling_rate=1,
    batch_size=batch_size,
)

# 将数据转换为 TensorFlow 张量
input_data = []
for batch in Dataset_input.as_numpy_iterator():
    input_data.extend(batch)
input_data_tensor = tf.convert_to_tensor(input_data)

#labeldata
label_data=capacity[past:]
Dataset_Label=tf.keras.preprocessing.timeseries_dataset_from_array(
    label_data,
    None,
    sequence_length=future,
    sampling_rate=1,
    batch_size=batch_size
)
output_data = []
for batch in Dataset_Label.as_numpy_iterator():
    output_data.extend(batch)
output_data_tensor = tf.convert_to_tensor(output_data)

#将batchsize 给他对应起来做乱序
np.random.seed(4)
shuffled_indices=np.random.shuffle(tf.range(len(output_data_tensor)))
shuffled_input_data = tf.gather(input_data_tensor, shuffled_indices)
shuffled_output_data = tf.gather(output_data_tensor, shuffled_indices)









###LSTM model training###
################################################################################################################################
#input一定是这样的,5个神经元，每个神经元能放N个特征进去。 这里单纯就是塑框架形状，还没有训练.
learning_rate = 0.001

multi_lstm_model = tf.keras.models.Sequential([
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

#就才是训练核心步骤
history = multi_lstm_model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback]
)
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
predictions = multi_lstm_model.predict(x_pre_reshaped )
#print(predictions)
#调最后一个window, 画图看一下就行了
true_values = Prediction_data[0:9]
#print(true_values)
predicted_values = predictions[0]
#print(predicted_values)

# 绘制对比图
plt.scatter(range(1,len(true_values)+1), true_values, label='True Capacity')
plt.scatter([6,7,8,9], predicted_values,label='Predicted Capacity')
plt.title('True Capacity vs Predicted Capacity')
plt.xlabel('Predicted Cycles')
plt.ylabel('Capacity')
plt.legend()
plt.show()


