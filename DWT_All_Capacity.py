import pywt
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
#B_0005
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0005.mat')
RD=mat_data['B0005']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
capacity_B0005=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0005.append(find_capacity)


import numpy as np
import matplotlib.pyplot as plt
import pywt


#定义小波且扫描
coeffs_B0005=pywt.dwt(capacity_B0005,'db1')

#扫描完成且构建 两个系数
cA_B0005, cD_B0005 = coeffs_B0005
# 把detail coeffcient  清0
cD_modified_B0005 = np.zeros((84,))
#cD_modified=pywt.threshold(cD,value=0.001, mode='soft')
#print(len(cD))
#print(type(cD))
#print(cD_modified)
#把两个系数提取做重构
reconstructed_signal_B0005 = pywt.idwt(cA_B0005,cD_modified_B0005, 'db1')



# 绘制原始信号
#    plt.subplot(4, 1, 1)
#   plt.plot(capacity_B0005)
#    plt.title('Original Signal_B0005')

# 绘制逼近系数
#    plt.subplot(4, 1, 2)
#    plt.plot(np.arange(len(cA)),cA, color='blue')
#    plt.title('Approximation Coefficients')
#    plt.xlabel('Time')
#    plt.ylabel('Coefficient Value')

# 绘制细节系数
#    plt.subplot(4, 1, 3)
#    plt.plot(np.arange(len(cD)), cD_modified, color='red')
#    plt.title('Detail Coefficients')
#    plt.xlabel('Time')
#    plt.ylabel('Coefficient Value')


# 绘制重构信号
#    plt.subplot(4, 1, 4)
#    plt.plot(reconstructed_signal)
#    plt.title('Reconstructed Signal_B0005')
#    plt.tight_layout()
#    plt.show()





#B_0006
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0006.mat')
RD=mat_data['B0006']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
capacity_B0006=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0006.append(find_capacity)

coeffs_B0006=pywt.dwt(capacity_B0006,'db1')

cA_B0006, cD_B0006 = coeffs_B0006

cD_modified_B0006 = np.zeros((84,))

reconstructed_signal_B0006 = pywt.idwt(cA_B0006,cD_modified_B0006, 'db1')










#B_0007
mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0007.mat')
RD=mat_data['B0007']
RD=RD[0][0][0][0]
size=range(RD.shape[0])
capacity_B0007=[]
for i in size:
    if  RD[i][0][0]=='discharge':
        find_capacity=RD[i][3][0][0][-1][0][0]
        capacity_B0007.append(find_capacity)

coeffs_B0007=pywt.dwt(capacity_B0007,'db1')

cA_B0007, cD_B0007 = coeffs_B0007

cD_modified_B0007 = np.zeros((84,))

reconstructed_signal_B0007 = pywt.idwt(cA_B0007,cD_modified_B0007, 'db1')






if __name__ == "__main__":


#B_0018
    mat_data = loadmat('C:/Users/zheng/Desktop/MA/FY08Q4/B0018.mat')
    RD=mat_data['B0018']
    RD=RD[0][0][0][0]
    size=range(RD.shape[0])
    capacity_B0018=[]
    for i in size:
        if  RD[i][0][0]=='discharge':
            find_capacity=RD[i][3][0][0][-1][0][0]
            capacity_B0018.append(find_capacity)


