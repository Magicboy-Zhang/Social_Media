import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
此函数用于生成仅通过两个特征进行分类的分类器的决策边界图像

参数：
    X_train:        (pandas Dataframe)  can only has 2 feature
    X_test:         (pandas Dataframe)  can only has 2 feature
    y_train:        (pandas Dataframe)  training target (分类变量)
    y_test:         (pandas Dataframe)  test target (分类变量)
    clf:            (model object)  classifier 
    show:           (string)        either 'train' or 'test', depends on what data points you want to observe
    title:          (string)        title of the plot
'''
def plot_decision_boundary2F2D(X_train, X_test, y_train, y_test, clf, show, title):

    '''
    拟合模型
    '''
    # 分类器拟合数据
    clf.fit(X_train.values, y_train.values.ravel())
    
    '''
    计算空间中所有点的预测类别
    '''
    # 得到x轴和y轴的范围
    x_min, x_max = X_train['sepal length (cm)'].min() - 0.2, X_train['sepal length (cm)'].max() + 0.2
    y_min, y_max = X_train['petal width (cm)'].min() - 0.2, X_train['petal width (cm)'].max() + 0.2

    # 创建网格
    #   np.meshgrid 用于生成二维坐标矩阵。np.arange 用于生成从 x_min 到 x_max，
    #   以及从 y_min 到 y_max 的均匀间隔的值。xx 和 yy 分别是 x 轴和 y 轴上的坐标矩阵。
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 预测决策边界
    #   将网格中的坐标点展平为一维数组，通过 np.c_ 拼接为一个二维数组，然后使用 KNN 
    #   模型 knn 进行预测，得到预测结果 Z。最后，通过 reshape 将一维数组转换为与网格形状相同的二维数组。
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    '''
    作图
    '''
    if show == 'train':
        # 绘制数据点
        sns.scatterplot(x = X_train.iloc[:, 0].values, y = X_train.iloc[:, 1].values, hue = y_train.iloc[:, 0].values, 
                    palette = 'viridis', edgecolors = 'k', marker = 'o')
    else:
        sns.scatterplot(x = X_test.iloc[:, 0].values, y = X_test.iloc[:, 1].values, hue = y_test.iloc[:, 0].values, 
                    palette = 'viridis', edgecolors = 'k', marker = 'o')

    # 绘制决策边界
    # 使用 contourf 函数绘制决策边界，通过填充等高线图的方式呈现分类区域。
    plt.contourf(xx, yy, Z, alpha = 0.3, cmap = 'viridis')

    # 设置标题和轴标签
    plt.title(title)
    plt.legend(title = 'label', loc = 'lower right')
    plt.xlabel(X_train.columns.tolist()[0])
    plt.ylabel(X_train.columns.tolist()[1])

    plt.show()

    '''
此函数用于生成指定K的KNN拟合曲线

参数：
    X_train:        (pandas Dataframe)  can only has one feature
    X_test:         (pandas Dataframe)  can only has one feature
    y_train:        (pandas Dataframe)  training target (连续变量)
    y_test:         (pandas Dataframe)  test target (连续变量)
    clf:            (model object)  classifier 
    show:           (string)        either 'train' or 'test', depends on what data points you want to observe
    curve_name:     (string)        name of the curve, shown on the plot
    title:          (string)        title of the plot
'''
def plot_fitted_curve1F2D(X_train, X_test, y_train, y_test, reg, show, curve_name, title):
    
    '''
    拟合模型
    '''
    # 回归器拟合数据(大部分回归器都期望特征是二维数组)
    reg.fit(X_train.values.reshape(-1, 1), y_train.values.ravel())

    '''
    计算所有X的预测值
    '''
    # 得到x的范围
    x_min, x_max = X_train['MedInc'].min() - 0.2, X_train['MedInc'].max() + 0.2

    # 生成覆盖该范围的x的数列
    xx = np.arange(x_min, x_max, 0.01)

    # 对整个x的数列进行预测并保存结果
    y_pred = reg.predict(xx.reshape(-1, 1))

    '''
    绘制拟合函数曲线
    '''
    if show == 'train':
        # 图中的点为训练集的数据点
        sns.scatterplot(x = X_train.iloc[:, 0].values, y = y_train.iloc[:, 0].values)

    else:
        # 图中的点为测试集的数据点
        sns.scatterplot(x = X_test.iloc[:, 0].values, y = y_test.iloc[:, 0].values)

    # 绘制模型拟合的函数曲线
    plt.plot(xx, y_pred, label = curve_name, color = 'red', linewidth = 1)

    # 设置图形属性
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.show()