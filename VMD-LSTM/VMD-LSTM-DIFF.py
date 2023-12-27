# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:35:51 2023

@author: 大月子
"""


# In[1]:


# 调用相关库
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas import DataFrame
from pandas import concat
import numpy
import os


# In[2]:


# 加载数据集
dataset = pd.read_csv(r"C:\Users\18732\Desktop\论文\台湾1935-2023.csv")
fratures_Corrected_irradiance = [
   'latitude', 'longitude', 'mg','diff'
]
values = dataset[fratures_Corrected_irradiance].values


# In[3]:


import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['font.sans-serif']=['Simhei']  #解决画图中文不显示问题
plt.figure(figsize=(6,4),dpi=600)
#添加网格线,linestyle表示绘制网格线的形式，alpha表示透明度
# plt.grid(True,linestyle="--",alpha=0.5)  
data10=dataset['diff'].values
# x=range(1, len(mean_pre_test) + 1)
# plt.xticks(x[::50])
plt.tick_params(labelsize=10)  #改变刻度字体大小
plt.plot(data10, color='black',linewidth=1)
plt.rcParams.update({'font.size': 10})  #改变图例里面的字体大小
plt.xlabel("Sample",fontsize=10)
plt.ylabel("Diff",fontsize=10)
plt.show()


# In[4]:


data=values
close_diff = data[:,3] # 取出对应列
close_diff.shape


# In[5]:


import numpy as np
import math

def VMD(signal, alpha, tau, K, DC, init, tol):
    # ---------------------
    #  signal  - the time domain signal (1D) to be decomposed
    #  alpha   - the balancing parameter of the data-fidelity constraint
    #  tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    #  K       - the number of modes to be recovered
    #  DC      - true if the first mode is put and kept at DC (0-freq)
    #  init    - 0 = all omegas start at 0
    #                     1 = all omegas start uniformly distributed
    #                     2 = all omegas initialized randomly
    #  tol     - tolerance of convergence criterion; typically around 1e-6
    #
    #  Output:
    #  -------
    #  u       - the collection of decomposed modes
    #  res     - residual
    #  u_hat   - spectra of the modes
    #  omega   - estimated mode center-frequencies
    

    
    # Period and sampling frequency of input signal
    save_T=len(signal)
    fs=1/float(save_T)

    # extend the signal by mirroring
    T=save_T
    # print(T)
    f_mirror=np.zeros(2*T)
    #print(f_mirror)
    f_mirror[0:T//2]=signal[T//2-1::-1]
    # print(f_mirror)
    f_mirror[T//2:3*T//2]= signal
    # print(f_mirror)
    f_mirror[3*T//2:2*T]=signal[-1:-T//2-1:-1]
    # print(f_mirror)
    f=f_mirror
    # print('f_mirror')
    # print(f_mirror)
    # print('-------')

    # Time Domain 0 to T (of mirrored signal)
    T=float(len(f))
    # print(T)
    t=np.linspace(1/float(T),1,int(T),endpoint=True)
    # print(t)

    # Spectral Domain discretization
    freqs=t-0.5-1/T
    # print(freqs)
    # print('-----')
    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N=500

    # For future generalizations: individual alpha for each mode
    Alpha=alpha*np.ones(K,dtype=complex)
    # print(Alpha.shape)
    # print(Alpha)
    # print('-----')

    # Construct and center f_hat
    f_hat=np.fft.fftshift(np.fft.fft(f))
    # print('f_hat')
    # print(f_hat.shape)
    # print(f_hat)
    # print('-----')
    f_hat_plus=f_hat
    f_hat_plus[0:int(int(T)/2)]=0
    # print('f_hat_plus')
    # print(f_hat_plus.shape)
    # print(f_hat_plus)
    # print('-----')
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus=np.zeros((N,len(freqs),K),dtype=complex)
    # print('u_hat_plus')
    # print(u_hat_plus.shape)
    # print(u_hat_plus)
    # print('-----')


    # Initialization of omega_k
    omega_plus=np.zeros((N,K),dtype=complex)
    # print('omega_plus')
    # print(omega_plus.shape)
    # print(omega_plus)
                        
    if (init==1):
        for i in range(1,K+1):
            omega_plus[0,i-1]=(0.5/K)*(i-1)
    elif (init==2):
        omega_plus[0,:]=np.sort(math.exp(math.log(fs))+(math.log(0.5)-math.log(fs))*np.random.rand(1,K))
    else:
        omega_plus[0,:]=0

    if (DC):
        omega_plus[0,0]=0

    # print('omega_plus')
    # print(omega_plus.shape)
    # print(omega_plus)

    # start with empty dual variables
    lamda_hat=np.zeros((N,len(freqs)),dtype=complex)

    # other inits
    uDiff=tol+2.2204e-16 #updata step
    # print('uDiff')
    # print(uDiff)
    # print('----')
    n=1 #loop counter
    sum_uk=0 #accumulator

    T=int(T)


    # ----------- Main loop for iterative updates

    while uDiff > tol and n<N:
        # update first mode accumulator
        k=1
        sum_uk = u_hat_plus[n-1,:,K-1]+sum_uk-u_hat_plus[n-1,:,0]
    #     print('sum_uk')
    #     print(sum_uk)
        #update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
    #     print('u_hat_plus')
    #     print(u_hat_plus.shape)
    #     print(u_hat_plus[n,:,k-1])
    #     print('-----')
        
        

        #update first omega if not held at 0
        if DC==False:
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T,k-1])))


        for k in range(2,K+1):

            #accumulator
            sum_uk=u_hat_plus[n,:,k-2]+sum_uk-u_hat_plus[n-1,:,k-1]
    #         print('sum_uk'+str(k))
    #         print(sum_uk)


            #mode spectrum
            u_hat_plus[n,:,k-1]=(f_hat_plus-sum_uk-lamda_hat[n-1,:]/2)/(1+Alpha[k-1]*np.square(freqs-omega_plus[n-1,k-1]))
    #         print('u_hat_plus'+str(k))
    #         print(u_hat_plus[n,:,k-1])
            
            #center frequencies
            omega_plus[n,k-1]=np.dot(freqs[T//2:T],np.square(np.abs(u_hat_plus[n,T//2:T,k-1])).T)/np.sum(np.square(np.abs(u_hat_plus[n,T//2:T:,k-1])))
    #         print('omega_plus'+str(k))
    #         print(omega_plus[n,k-1])
        #Dual ascent
    #     print(u_hat_plus.shape)
        lamda_hat[n,:]=lamda_hat[n-1,:]+tau*(np.sum(u_hat_plus[n,:,:],axis=1)-f_hat_plus)
    #     print('lamda_hat'+str(n))
    #     print(lamda_hat[n,:])

        #loop counter
        n=n+1

        #converged yet?
        uDiff=2.2204e-16

        for i in range(1,K+1):
            uDiff=uDiff+1/float(T)*np.dot(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1],(np.conj(u_hat_plus[n-1,:,i-1]-u_hat_plus[n-2,:,i-1])).conj().T)

            
        
        uDiff=np.abs(uDiff)
        # print('uDiff')
        # print(uDiff)

        
    # ------ Postprocessing and cleanup

    # discard empty space if converged early

    N=np.minimum(N,n)
    omega = omega_plus[0:N,:]

    # Signal reconstruction
    u_hat = np.zeros((T,K),dtype=complex)
    u_hat[T//2:T,:]= np.squeeze(u_hat_plus[N-1,T//2:T,:])
    # print('u_hat')
    # print(u_hat.shape)
    # print(u_hat)
    u_hat[T//2:0:-1,:]=np.squeeze(np.conj(u_hat_plus[N-1,T//2:T,:]))
    u_hat[0,:]=np.conj(u_hat[-1,:])
    # print('u_hat')
    # print(u_hat)
    u=np.zeros((K,len(t)),dtype=complex)

    for k in range(1,K+1):
        u[k-1,:]= np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k-1])))


    # remove mirror part 
    u=u[:,T//4:3*T//4]
    u=np.real(u)
    
    res=signal-u.sum(axis=0)
    # print(u_hat.shape)
    #recompute spectrum
    u_hat = np.zeros((T//2,K),dtype=complex)

    for k in range(1,K+1):
        u_hat[:,k-1]=np.fft.fftshift(np.fft.fft(u[k-1,:])).conj().T
        
        
    return (u,u_hat,omega)


# In[6]:


# In[] vmd分解-直接设置参数
alpha = 1000  # 带宽限参数 频带限制
k = 6    # 模态分解数量
tau = 0  # 对噪声的容忍程度 一般取0 允许有误差
DC = 0     # 一般取0
init = 1     #   每一个的中心频率 0所有中心频率为0 1 均匀分布 2 随机
tol = 1e-7     # 收敛精度 
u_diff,u_hat_diff,omega_diff=VMD(close_diff, alpha, tau, k, DC, init, tol)
 # u对应时间序列 u_hat频谱 omega每个模态中心频率


# In[7]:


# u_mg


# In[10]:


from matplotlib import rcParams


import seaborn as sns

plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['font.family'] = ['sans-serif']
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['font.sans-serif']=['Simhei']  #解决画图中文不显示问题
plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
plt.figure(figsize=(10,8), dpi=500)
for i in range(k):
    plt.rcParams['font.family'] = ['sans-serif']

   
    plt.subplot(k,1,i+1)
    plt.plot(u_diff[i,:],color = 'black',linewidth=1)
    plt.xlabel("sample")
    plt.ylabel('IMF{}'.format(i+1))
# plt.savefig('新疆VMD分解结果.png')


# In[11]:

# 定义LSTM
def implement_Bigru(train_X, train_y):
    inputs=Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm=LSTM(128, activation='selu',return_sequences=False)(inputs)
    outputs = Dense(1)(lstm)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse',optimizer='Adam')
    model.summary()#展示模型结构
    return model


# In[12]:


### 构建时间序列特征集
def series_to_supervised(data, n_in=3, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg   #


# In[13]:


i = 1
svr = []
prediction_test = []
actual_test = []

for imf in u_diff:
    imf=imf.reshape(-1,1)
    imf=pd.DataFrame(imf)
    data_1=pd.DataFrame(data[:,(0,1,2)]) # 取出除对应列以外的其他列
    data_=pd.concat((imf,data_1), axis=1)
    data_=data_.values
    data_ = data_.astype('float32')
    
    # 构建成监督学习问题
    n_in=3 # 步长，试一下别的
    n_out=1
    n_vars=4
# 构建成监督学习问题
    reframed = series_to_supervised(data_, n_in, n_out)  # 预测时间步长为3 ，输出时间步1  ，即由t-15,t-14,....t-1预测t
    #取出保留的变量
    contain_vars = []
    for i in range(1, n_in+1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1,n_vars+1)]  
    data3 = reframed [ contain_vars + ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1,n_out)]]
    values = data3.values
    n_train_hours =int(len(values)*0.7)  # 区分训练集和测试集
    train = values[:n_train_hours, :]  # n_train行，所有列
    test = values[n_train_hours:, :]   
    # 归一化，可以试试最大最小归一
    scaler =  StandardScaler()
    train = scaler.fit_transform(train)
    test =  scaler.fit_transform(test)
    # 把数据分为输入和输出
    train_X, train_y = train[:, :n_in*n_vars], train[:, n_in*n_vars:]
    test_X, test_y = test[:, :n_in*n_vars], test[:, n_in*n_vars:]
    # 把输入重塑成3D格式 [样例，时间步， 特征]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    test_X = test_X.reshape((test_X.shape[0],n_in, n_vars))
 

    tmp = implement_Bigru(train_X, train_y)
    history = tmp.fit(train_X, train_y, epochs=80, batch_size=30, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    svr.append(tmp)
    
    
    
    # 作出预测
    yhat = tmp.predict(test_X)
    # 反向缩放预测值  测试集
    yhat = np.repeat(yhat,n_in*n_vars+n_out, axis=-1)
    inv_yhat=scaler.inverse_transform(np.reshape(yhat,(len(yhat),n_in*n_vars+n_out)))[:,0]
    prediction_test.append(inv_yhat)
    # 反向缩放实际值  测试集
    y = np.repeat(test_y,n_in*n_vars+n_out, axis=-1)
    inv_y=scaler.inverse_transform(np.reshape(y,(len(test_y),n_in*n_vars+n_out)))[:,0]
    actual_test.append(inv_y)

    

    
    
i=i+1


# In[14]:


mean_pre_test = []                    

for i in range(0,len(prediction_test[0])):
    sum = 0
    for j in range(0,len(prediction_test)):
        sum = sum + prediction_test[j][i]
    
#     mean = sum/len(prediction_test) 
    mean_pre_test.append(sum)
        
mean_pre_test        


# In[15]:


mean_actual_test = []

for i in range(0,len(actual_test[0])):
    sum = 0
    for j in range(0,len(actual_test)):
        sum = sum + actual_test[j][i]
    
#     mean = sum/len(actual_test)
    mean_actual_test.append(sum)
        
mean_actual_test    


# In[16]:


# 计算RMSE
rmse = sqrt(mean_squared_error(mean_actual_test, mean_pre_test))
print('Test RMSE: %.7f' % rmse)
print('Test MAE: %.7f' % mean_absolute_error(mean_actual_test, mean_pre_test))
print('Test R2: %.7f' % r2_score(mean_actual_test, mean_pre_test))
# print('Test R2: %.7f' % r2_score(inv_y,inv_yhat))


# In[18]:


import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['font.sans-serif']=['Simhei']  #解决画图中文不显示问题
plt.figure(figsize=(6,4),dpi=600)
#添加网格线,linestyle表示绘制网格线的形式，alpha表示透明度
# plt.grid(True,linestyle="--",alpha=0.5)  
x=range(1, len(mean_pre_test) + 1)
plt.xticks(x[::50])
plt.tick_params(labelsize=10)  #改变刻度字体大小
plt.plot(x, close_diff[1190-len(mean_pre_test):], marker='*',markersize='1',color='g',linewidth=1, label='True diff')
plt.plot(x, mean_pre_test, marker='s',markersize='1', color='red',linestyle="--",linewidth=1,label='Predicting diff')
plt.rcParams.update({'font.size': 10})  #改变图例里面的字体大小
plt.legend(loc='upper right')
plt.xlabel("Sample",fontsize=10)
plt.ylabel("Diff",fontsize=10)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




