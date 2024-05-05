import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.io
from scipy.io import loadmat
from DWT_All_Capacity import reconstructed_signal_B0005
from DWT_All_Capacity import reconstructed_signal_B0006
from DWT_All_Capacity import reconstructed_signal_B0007
#Input Data
capacity_B0005=reconstructed_signal_B0005
capacity_B0006=reconstructed_signal_B0006
capacity_B0007=reconstructed_signal_B0007
###全数据集做normalization
def min_max_normalization(Data):
    min_val = min(Data)
    max_val = max(Data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in Data]
    return normalized_data
capacity_B0005=min_max_normalization(capacity_B0005[:])
capacity_B0006=min_max_normalization(capacity_B0006[:])
capacity_B0007=min_max_normalization(capacity_B0007[:])
data = capacity_B0005+capacity_B0006
#我现在已经有一个连续的336组数据，我现在要做的就是用以 5+4 的框滑过去整理出train_input 和 train_label
past=5
future=4
windowsize=past+future
batch_size=len(data)-windowsize
#整理input
input_Dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    data,
    None,
    sequence_length=past,
    sampling_rate=1,
    batch_size=past,
)
#还是得截取一下batchsize的长度
input_data = []
for batch in input_Dataset.as_numpy_iterator():
    input_data.extend(batch)
#手动截成327个数据
train_input = input_data[:batch_size]
train_input=tf.expand_dims(train_input, axis=-1)
print(train_input.shape)




#整理label
label_data=data[past:]
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
#手动截成327个数据
train_label=label_data[:batch_size]
train_label=tf.expand_dims(train_label, axis=-1)
print(train_label.shape)

#model
learning_rate = 0.002
num_features=1
multi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(None, num_features)),
    tf.keras.layers.Dense(future * num_features, kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([future, num_features])
])

multi_lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=tf.keras.losses.MeanSquaredError())
multi_lstm_model.summary()

#train
epochs=70
history = multi_lstm_model.fit(
    train_input,
    train_label,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
)


#test
test_input = capacity_B0007[:past]
test_input = np.array(test_input)
test_input_reshaped = test_input.reshape(1, past, num_features)
prediction_data = multi_lstm_model.predict(test_input_reshaped)
true_values=capacity_B0007[:windowsize]

plt.scatter(range(1,len(true_values)+1), true_values, label='True Capacity')
plt.scatter([6,7,8,9], prediction_data,label='Predicted Capacity')
plt.title('True Capacity vs Predicted Capacity')
plt.xlabel('Predicted Cycles')
plt.ylabel('Capacity')
plt.legend()
plt.show()
