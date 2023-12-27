# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:48:05 2023

@author: 大月子
"""

# 调用相关库
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas import DataFrame
from pandas import concat
import numpy
from keras.callbacks import LearningRateScheduler
import os


# In[2]:


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
    return agg


# In[3]:


# 加载数据集
dataset = pd.read_csv(r"C:\Users\18732\Desktop\论文\台湾1935-2023.csv")
fratures_Corrected_irradiance = [
   'latitude', 'longitude', 'mg','diff'
]
values = dataset[fratures_Corrected_irradiance].values


# In[4]:


values.shape


# In[5]:


# 确保所有数据是浮动的
values = values.astype('float32')
n_in=3
n_out=1
n_vars=4  #特征数3个

# 构建成监督学习问题
reframed = series_to_supervised(values, n_in, n_out)  # 预测时间步长为3 ，输出时间步1  ，即由t-3,t-2,....t-1预测t


# In[6]:


reframed 


# In[7]:


#取出保留的变量
contain_vars = []
for i in range(1, n_in+1):
    contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1,n_vars+1)]  
data = reframed [ contain_vars + ['var3(t)'] + [('var3(t+%d)' % (j)) for j in range(1,n_out)]]


# In[8]:


# 把数据集分为训练集和测试集
values = data.values
n_train_hours =int(len(values)*0.7)   # 70%数据训练
train = values[:n_train_hours, :]  # 前7000组数据训练 验证
test = values[n_train_hours:, :]   # 后1761组预测


# In[9]:


# 标准化
scaler = StandardScaler()
train = scaler.fit_transform(train)
test =  scaler.fit_transform(test)


# In[10]:


# 把数据分为输入和输出
train_X, train_y = train[:, :n_in*n_vars], train[:, n_in*n_vars:]
test_X, test_y = test[:, :n_in*n_vars], test[:, n_in*n_vars:]


# In[11]:


# 把输入重塑成3D格式 [样例，时间步， 特征]
train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
test_X = test_X.reshape((test_X.shape[0],n_in, n_vars))
print("train_X.shape:%s train_y.shape:%s test_X.shape:%s test_y.shape:%s" %(train_X.shape, train_y.shape, test_X.shape, test_y.shape))


# In[12]:


train_X


# In[13]:


def lstm_model():
#建立模型
    inputs=Input(shape=(train_X.shape[1], train_X.shape[2]))
    lstm=LSTM(64, activation='selu',return_sequences=False)(inputs)
    #dense=Dropout(dropout)(rnn)#droupout层
    outputs = Dense(1)(lstm)
    model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='mse')
    model.compile(loss='mse',optimizer='Adam')
    model.summary()#展示模型结构
    return model



model = lstm_model()#建立模型
history = model.fit(train_X, train_y, epochs=80, batch_size=30, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)


# In[14]:


# 作出预测
yhat = model.predict(test_X)  #357行1列
yhat = np.repeat(yhat,n_in*n_vars+n_out, axis=-1)
#inv_yhat=scaler.inverse_transform(yhat)[:,2]
inv_yhat=scaler.inverse_transform(np.reshape(yhat,(len(yhat),n_in*n_vars+n_out)))[:,2]
y = np.repeat(test_y,n_in*n_vars+n_out, axis=-1)
#inv_y=scaler.inverse_transform(y)[:,2]
inv_y=scaler.inverse_transform(np.reshape(y,(len(test_y),n_in*n_vars+n_out)))[:,2]


# In[15]:


# 计算RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.7f' % rmse)
print('Test MAE: %.7f' % mean_absolute_error(inv_y, inv_yhat))
print('Test R2: %.7f' % r2_score(inv_y,inv_yhat))


# In[17]:


import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['font.sans-serif']=['Simhei']  #解决画图中文不显示问题
plt.figure(figsize=(6,4),dpi=600)
#添加网格线,linestyle表示绘制网格线的形式，alpha表示透明度
# plt.grid(True,linestyle="--",alpha=0.5)  
x=range(1, len(inv_yhat) + 1)
plt.xticks(x[::50])
plt.tick_params(labelsize=10)  #改变刻度字体大小
plt.plot(x, inv_y, marker='*',markersize='1',color='g',linewidth=1, label='True magnitude')
plt.plot(x, inv_yhat, marker='s',markersize='1', color='red',linestyle="--",linewidth=1,label='Predicting magnitude')
plt.rcParams.update({'font.size': 10})  #改变图例里面的字体大小
plt.legend(loc='upper right')
plt.xlabel("Sample ",fontsize=10)
plt.ylabel("magnitude",fontsize=10)
plt.legend()
# plt.xlim(xmin=600,xmax=700)  #显示600-1000的值   局部放大有利于观察
plt.show()