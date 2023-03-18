from sklearn.metrics import precision_recall_curve,auc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import csv
import pandas as pd




if __name__=='__main__':
    data = pd.read_csv(r'E:\jiangshan\U-net\Pytorch-UNet\data\evaluate_demo\pred_results\test.csv')  # 读取评价参数
    # print(data.shape)#[行数，列数]
    # data.values可以查看DataFrame里的数据值，返回的是一个数组。
    #读取到的数据返回一个DataFrame,类似excel，是一种二维表
    #测试PR曲线方法
    pr_xdata = []
    pr_ydata = []
    # xdata = data.ix[:,'列名']   #将csv中列名为“列名1”的列存入xdata数组中
    # ydata = data.ix[:,'列名']   #将csv中列名为“列名2”的列存入ydata数组中
    # loc是根据行名，iloc是根据行的索引。如果查看多行要多嵌套一个中括号。
    pr_ydata = data.iloc[:,1] #P,num处写要读取的列序号，0为csv文件第一列
    pr_xdata = data.iloc[:,2]   #R
    list_precision=data.values[:,1]
    list_recall=data.values[:,2]

    plt.plot(pr_xdata,pr_ydata,'bo-',label=u'',linewidth=1)
    plt.title(u"PR_curve",size=10)   #设置表名为“表名”
    plt.legend()
    plt.xlabel(u'Recall',size=10)   #设置x轴名为“x轴名”
    plt.ylabel(u'Precision',size=10)   #设置y轴名为“y轴名”
    plt.show()

    #AP计算
    # list1=[0,0.5,1]#横坐标
    # list2=[0,1,0]
    # print(auc(list1,list2))
    AP=auc(list_recall,list_precision)#auc方法计算横纵坐标构成曲线与轴间的面积
    print('AP:', AP)

    #Iou曲线：



    #ROC曲线
    roc_xdata= data.iloc[:,5]#FPR
    roc_ydata= data.iloc[:,6]#TPR
    fpr_list=data.values[:,5]
    tpr_list=data.values[:,6]
    plt.plot(roc_xdata, roc_ydata, 'bo-', label=u'', linewidth=1)
    plt.title(u"ROC_curve", size=10)  # 设置表名为“表名”
    plt.legend()
    plt.xlabel(u'FPR', size=10)  # 设置x轴名为“x轴名”
    plt.ylabel(u'TPR', size=10)  # 设置y轴名为“y轴名”
    plt.show()
    #auc计算
    AUC = auc(fpr_list,tpr_list)  # auc方法计算横纵坐标构成曲线与轴间的面积
    print('AUC:', AUC)

    #**F1值** = 准确率 * 召回率 * 2 / (准确率 + 召回率)
    #平衡点F1指P=R时的情况






