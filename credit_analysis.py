#!/usr/bin/env python
# coding: utf-8

# In[1]:


#使用逻辑回归对信用卡欺诈进行分类
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_curve,accuracy_score,precision_score,recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#混淆矩阵可视化
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues) :
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[23]:


#显示模型评估结果
def show_metrics(y_test,y_predict):
    a=accuracy_score(y_test,y_predict)
    b=precision_score(y_test,y_predict)
    c=recall_score(y_test,y_predict)
    d=(2*b*c)/(b+c)
    print('准确率：{:.3f}'.format(a))
    print('精确率：{:.3f}'.format(b))
    print('召回率：{:.3f}'.format(c))
    print('F1值：{:.3f}'.format(d))


# In[24]:


#绘制精确率-召回率曲线
def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率 曲线')
    plt.show()


# In[29]:


#数据加载
data=pd.read_csv('./creditcard.csv')


# In[30]:


data.head()


# In[31]:


#data中没有空值
data.info()


# In[32]:


#数据探索
print(data.describe())


# In[33]:


#绘制交易类别分布
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure()
sns.countplot(x='Class',data=data)
plt.title('类别分布')
plt.show()


# In[34]:


#显示总交易笔数、诈骗交易笔数
num=len(data)
num_fraud=len(data[data['Class']==1])
print('总交易笔数：',num)
print('诈骗交易笔数：',num_fraud)
print('诈骗交易比例：{:.6f}'.format(num_fraud/num))


# In[51]:


#正常交易与诈骗交易可视化
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(15,8))
bins=50
ax1.hist(data.Time[data.Class==1],bins=bins,color='deepskyblue')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class==0],bins=bins,color='deeppink')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')
plt.show()


# In[43]:


#对Amount进行数据规范化处理
data['Amount_Norm']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))


# In[9]:


#特征选择
y=data['Class']
x=data.drop(labels=['Time','Amount','Class'],axis=1)


# In[61]:


#数据集划分为训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=33)


# In[62]:


#逻辑回归分类
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)


# In[63]:


#计算、显示混淆矩阵
cm=confusion_matrix(y_test,y_predict)
class_names=[0,1]
plot_confusion_matrix(cm,classes=class_names,title='逻辑回归—混淆矩阵')


# In[64]:


#显示模型评估分数
show_metrics(y_test,y_predict)


# In[69]:


#预测样本的置信分数
y_score=clf.decision_function(x_test)


# In[71]:


#计算精确率，召回率，阈值用于可视化
precision,recall,thresholds=precision_recall_curve(y_test,y_score)
plot_precision_recall()


# In[ ]:




