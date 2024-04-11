import pywt
from scipy.io import loadmat
import matplotlib.pyplot as plt
#B_0005
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0005.mat')
RD=mat_data['B0005']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
print('size5',size)
capacity_B0005=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0005.append(find_capacity)



#from LSTM_B0005 import min_max_normalization
#capacity_B0005=min_max_normalization(capacity_B0005)

#看看db1 长什么样
import numpy as np
import matplotlib.pyplot as plt
import pywt



# 定义 DB1小波函数
wavelet = pywt.Wavelet('db1')
phi, psi, x = wavelet.wavefun()

# 绘制 Daubechies 小波函数
plt.plot(x, phi, label='Scaling Function (phi)')
plt.plot(x, psi, label='Wavelet Function (psi)')
plt.title('Daubechies Wavelet (db1)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()



#定义小波且扫描
coeffs_B0005=pywt.dwt(capacity_B0005,'db1')


#扫描完成且构建 两个系数
cA, cD = coeffs_B0005c
# 把cD 清0
#把两个系数提取做重构
reconstructed_signal = pywt.idwt(cA, cD, 'db1')

# 绘制原始信号
plt.subplot(4, 1, 1)
plt.plot(capacity_B0005)
plt.title('Original Signal_B0005')

# 绘制逼近系数
plt.subplot(4, 1, 2)
plt.plot(cA)
plt.title('Approximation Coefficients_B0005')

# 绘制细节系数
plt.subplot(4, 1, 3)
plt.plot(cD)
plt.title('Detail Coefficients_B0005')

# 绘制重构信号
plt.subplot(4, 1, 4)
plt.plot(reconstructed_signal)
plt.title('Reconstructed Signal_B0005')

plt.tight_layout()
plt.show()






#B_0006
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0006.mat')
RD=mat_data['B0006']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
print('size6',size)
capacity_B0006=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0006.append(find_capacity)

#B_0007
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0007.mat')
RD=mat_data['B0007']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
print('size7',size)
capacity_B0007=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0007.append(find_capacity)

#B_0018
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0018.mat')
RD=mat_data['B0018']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
print('size18',size)
capacity_B0018=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0018.append(find_capacity)


