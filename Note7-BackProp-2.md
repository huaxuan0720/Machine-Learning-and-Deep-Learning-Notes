---
title: 反向传播算法（二）之稍复杂的反向传播
date: 2019-05-12 21:27:17
tags: [机器学习 ,反向传播算法]
categories: 机器学习
toc: true
thumbnail: gallery/MachineLearning.jpg
---

##### 前言  
&emsp;&emsp;前面介绍了单层全连接层，并且没有使用激活函数，这种情况比较简单，这一篇文章打算简单介绍一下多个输出，以及使用激活函数进行非线性激活的情况。还是请注意：这里的所有推导过程都只是针对当前设置的参数信息，并不具有一般性，但是所有的推导过程可以推导到一般的运算，因此以下给出的并不是反向传播算法的严格证明，但是可以很好的帮助理解反向传播算法。
<!--more-->

##### 一、参数设置  
&emsp;&emsp;和前面一样，这里使用的是长度为3的行向量，即 $x = \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix}$，输出这里设置为长度为2的行向量，即 $\hat{y} = \begin{bmatrix} \hat{y}_1 & \hat{y}_2 \end{bmatrix}$。权值参数我们记为 $\omega$，偏置量我们记为 $b$，由于这里我们模拟的是进行分类操作，因此这里引入了一个非线性激活函数 $g$，为了方便我们进行求导，我们这里设置激活函数为sigmoid，即:
$$g(x) = \frac{1}{1 + e^{-x}}, g\prime(x) = g(x)(1 - g(x)) \tag{1}$$

&emsp;&emsp;有了上述的参数设置，我们可以有下面的式子：
$$
g(x \omega + b) = \hat{y} \tag{2}
$$
&emsp;&emsp;继续将式子展开，我们有：
$$
g(\begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix} \begin{bmatrix} \omega_{11} & \omega_{12} \\ \omega_{21} & \omega_{22} \\ \omega_{31} & \omega_{32} \\ \end{bmatrix} + \begin{bmatrix} b_1 & b_2\end{bmatrix}) = \begin{bmatrix} \hat{y}_1 & \hat{y}_2 \end{bmatrix} \tag{3}
$$

##### 三、首先不考虑激活函数  
&emsp;&emsp;我们首先不考虑激活函数，因此，我们可以暂时将结果记为 $a = \begin{bmatrix} a_1 & a_2 \end{bmatrix}$。于是，我们可以得到下面的式子：
$$
\begin{bmatrix} a_1 & a_2 \end{bmatrix} = \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix} \begin{bmatrix} \omega_{11} & \omega_{12} \\ \omega_{21} & \omega_{22} \\ \omega_{31} & \omega_{32} \\ \end{bmatrix} + \begin{bmatrix} b_1 & b_2\end{bmatrix} \tag{4}
$$
 &emsp;&emsp;将上面的公式(4)完全展开，可以得到下面的两个式子：
$$
a_1 = \omega_{11} x_1 + \omega_{21} x_2 + \omega_{31} x_3 + b_1 \tag{5}
$$
$$
a_2 = \omega_{12} x_1 + \omega_{22} x_2 + \omega_{32} x_3 + b_2 \tag{6}
$$
&emsp;&emsp;和前面的情况类似，我们可以对上面的两个式子中的参数求偏导数，于是，我们得到对于各个参数的偏导数计算公式如下：
$$
\frac{\partial a_1}{\partial \omega_{11}} = x_1, \frac{\partial a_1}{\partial \omega_{21}} = x_2, \frac{\partial a_1}{\partial \omega_{31}} = x_3, \frac{\partial a_1}{\partial b_1} = 1 \tag{7}
$$
$$
\frac{\partial a_2}{\partial \omega_{12}} = x_1, \frac{\partial a_2}{\partial \omega_{22}} = x_2, \frac{\partial a_2}{\partial \omega_{32}} = x_3, \frac{\partial a_2}{\partial b_2} = 1 \tag{8}
$$

&emsp;&emsp;以上就是现阶段的偏导数的计算公式。下一阶段我们将激活函数也考虑进来。

##### 四、将激活函数也考虑进来  
&emsp;&emsp;这一阶段我们考虑对 $a = \begin{bmatrix} a_1 & a_2 \end{bmatrix}$ 使用非线性激活函数激活，即我们有：
$$
\hat{y} = g(a) \tag{9}
$$
&emsp;&emsp;展开之后就变成：
$$
\begin{bmatrix} \hat{y}_1 & \hat{y}_2 \end{bmatrix} = g(\begin{bmatrix} a_1 & a_2 \end{bmatrix}) \tag{10}
$$
&emsp;&emsp;对应每一个元素，我们有：
$$
\hat{y}_1 = g(a_1), \hat{y}_2 = g(a_2) \tag{11}
$$
&emsp;&emsp;所以我们求得每一个 $\hat{y}_i$ 对 $a_i$ 的偏导数如下：
$$
\frac{\partial \hat{y}_1}{\partial a_1} = g\prime(a_1), \frac{\partial \hat{y}_2}{\partial a_2} = g\prime(a_2) \tag{12}
$$

##### 五、损失值定义  
&emsp;&emsp;和前面的情况类似，我们使用输出与目标值之间的差值的平方和作为最后的cost，即：
$$
C = cost = \sum(\hat{y}_i - y_i)^2 = (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2 \tag{13}
$$
&emsp;&emsp;根据上式，我们可以得到 $C$ 关于两个预测输出 $\hat{y}_1$，$\hat{y}_2$的偏导数：
$$
\frac{\partial C}{\partial \hat{y}_1} = 2 * (\hat{y}_1 - y_1), \frac{\partial C}{\partial \hat{y}_2} = 2 * (\hat{y}_2 - y_2) \tag{14}
$$

##### 六、综合  
&emsp;&emsp;前面所做的工作实际上是在一步一步求解每一个环节的偏导数公式，根据求导公式的链式法则（chain rule），我们可以得到以下的每一个参数（$\omega$，$b$）对于最后的cost的偏导数公式：
$$
\frac{\partial C}{\partial \omega_{11}} = \frac{\partial a_1}{\partial \omega_{11}} \cdot \frac{\partial \hat{y}_1}{\partial a_1} \cdot \frac{\partial C}{\partial \hat{y}_1} = x_1 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \tag{15}
$$
$$
\frac{\partial C}{\partial \omega_{21}} = \frac{\partial a_1}{\partial \omega_{21}} \cdot \frac{\partial \hat{y}_1}{\partial a_1} \cdot \frac{\partial C}{\partial \hat{y}_1} = x_2 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \tag{16}
$$
$$
\frac{\partial C}{\partial \omega_{31}} = \frac{\partial a_1}{\partial \omega_{31}} \cdot \frac{\partial \hat{y}_1}{\partial a_1} \cdot \frac{\partial C}{\partial \hat{y}_1} = x_3 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \tag{17}
$$
$$
\frac{\partial C}{\partial b_1} = \frac{\partial a_1}{\partial b_1} \cdot \frac{\partial \hat{y}_1}{\partial a_1} \cdot \frac{\partial C}{\partial \hat{y}_1} = g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \tag{18}
$$
$$
\frac{\partial C}{\partial \omega_{12}} = \frac{\partial a_2}{\partial \omega_{12}} \cdot \frac{\partial \hat{y}_2}{\partial a_2} \cdot \frac{\partial C}{\partial \hat{y}_2} = x_1 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \tag{19}
$$
$$
\frac{\partial C}{\partial \omega_{22}} = \frac{\partial a_2}{\partial \omega_{22}} \cdot \frac{\partial \hat{y}_2}{\partial a_2} \cdot \frac{\partial C}{\partial \hat{y}_2} = x_2 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \tag{20}
$$
$$
\frac{\partial C}{\partial \omega_{32}} = \frac{\partial a_2}{\partial \omega_{32}} \cdot \frac{\partial \hat{y}_2}{\partial a_2} \cdot \frac{\partial C}{\partial \hat{y}_2} = x_3 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \tag{21}
$$
$$
\frac{\partial C}{\partial b_2} = \frac{\partial a_2}{\partial b_2} \cdot \frac{\partial \hat{y}_2}{\partial a_2} \cdot \frac{\partial C}{\partial \hat{y}_2} = g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \tag{22}
$$
&emsp;&emsp;和前面一样，上面的公式已经可以用于进行梯度计算和反向传播了，但是上面的公式看上去不仅繁琐而且容易出错，因此，很有必要对上面的公式进行整理，以便我们用向量和矩阵进行表示和计算。  
&emsp;&emsp;我们将每个变量的梯度按照次序排好，首先是 $\omega$ 参数的第一列，如下：
$$
\begin{bmatrix} \frac{\partial C}{\partial \omega_{11}} \\ \frac{\partial C}{\partial \omega_{21}} \\ \frac{\partial C}{\partial \omega_{31}} \\ \end{bmatrix} = \begin{bmatrix} x_1 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \\ x_2 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \\ x_3 \cdot g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \end{bmatrix} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \begin{bmatrix} g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) \end{bmatrix} \tag{23}
$$
&emsp;&emsp;接着是 $\omega$ 参数的第二列，如下：
$$
\begin{bmatrix} \frac{\partial C}{\partial \omega_{12}} \\ \frac{\partial C}{\partial \omega_{22}} \\ \frac{\partial C}{\partial \omega_{32}} \\ \end{bmatrix} = \begin{bmatrix} x_1 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \\ x_2 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \\ x_3 \cdot g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \end{bmatrix} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \begin{bmatrix} g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \end{bmatrix} \tag{24}
$$
&emsp;&emsp;将两个矩阵结合在一起：
$$
\begin {aligned}
\begin{bmatrix} \frac{\partial C}{\partial \omega_{11}} & \frac{\partial C}{\partial \omega_{12}} \\ \frac{\partial C}{\partial \omega_{121}} & \frac{\partial C}{\partial \omega_{22}} \\ \frac{\partial C}{\partial \omega_{31}} & \frac{\partial C}{\partial \omega_{32}} \end{bmatrix} &= \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \begin{bmatrix} g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) & g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \end{bmatrix} \\\ &= x^T \begin{bmatrix} g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) & g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \end{bmatrix} \\\ &= x^T (\begin{bmatrix} g\prime(a_1) & g\prime(a_2) \end{bmatrix} \cdot * \begin{bmatrix} 2 \cdot (\hat{y}_1 - y_1) & 2 \cdot (\hat{y}_2 - y_2)\end{bmatrix})
\end{aligned} \tag{25}
$$
&emsp;&emsp;注：上面公式中的 $\cdot*$ 为向量（矩阵）点乘，即表示向量（矩阵）对应位置的数值分别相乘，和矩阵的相乘不同。  
&emsp;&emsp;最后化简如下:
$$
\frac{\partial C}{\partial \omega} = x^T (\begin{bmatrix} g\prime(a_1) & g\prime(a_2) \end{bmatrix} \cdot * \begin{bmatrix} 2 \cdot (\hat{y}_1 - y_1) & 2 \cdot (\hat{y}_2 - y_2)\end{bmatrix}) \tag{26}
$$
&emsp;&emsp;对于偏置量 $b$，我们计算梯度，然后进行整理，如下：（这里实际上进行了一定的简化，当我们设定偏置量为一个一维数组时，我们需要对下面的结果在列方向上取均值，以保证最后的结果可以和偏置量进行维度上的匹配。）
$$
\begin{aligned}
\frac{\partial C}{\partial b}  &= \begin{bmatrix}  \frac{\partial C}{b_1} & \frac{\partial C}{b_2} \end{bmatrix} \\\ &= \begin{bmatrix} g\prime(a_1) \cdot 2 \cdot (\hat{y}_1 - y_1) & g\prime(a_2) \cdot 2 \cdot (\hat{y}_2 - y_2) \end{bmatrix} \\\ &= \begin{bmatrix} g\prime(a_1) & g\prime(a_2) \end{bmatrix} \cdot * \begin{bmatrix} 2 \cdot (\hat{y}_1 - y_1) & 2 \cdot (\hat{y}_2 - y_2)\end{bmatrix}
\end{aligned} \tag{27}
$$

&emsp;&emsp;以上就是单层全连接层，使用激活函数激活的梯度下降公式。


##### 七、代码  
&emsp;&emsp;这里我使用了两个训练样本，偏置量设定为一个1-D的数组，因此在更新参数时，需要对返回的结果取均值。详细请见backward函数。
```python
import numpy as np

param = {}
nodes = {}

learning_rate = 0.1


def sigmoid(x):
    return 1.0 / (1. + np.exp(- x))


def sigmoid_gradient(x):
    sig = sigmoid(x)
    return sig * (1. - sig)


def cost(y_pred, y):
    return np.sum((y_pred - y) ** 2)


def cost_gradient(y_pred, y):
    return 2 * (y_pred - y)


def forward(x):
    nodes['matmul'] = np.matmul(x, param['w'])
    nodes['bias'] = nodes['matmul'] + param['b']
    nodes['sigmoid'] = sigmoid(nodes['bias'])
    return nodes['sigmoid']


def backward(x, y_pred, y):
    matrix = np.multiply(sigmoid_gradient(nodes['bias']), cost_gradient(y_pred, y))
    matrix2 = np.mean(matrix, 0, keepdims=False)

    param['w'] -= learning_rate * np.matmul(np.transpose(x), matrix)
    param['b'] -= learning_rate * matrix2


def setup():
    x = np.array([[1., 2., 3.],
                  [3., 2., 1.]])
    y = np.array([[1., 0.],
                  [0., 1.]])

    param['w'] = np.array([[.1, .2], [.3, .4], [.5, .6]])
    param['b'] = np.array([0., 0.])

    for i in range(1000):
        y_pred = forward(x)
        backward(x, y_pred, y)
        print("梯度下降前：", y_pred, "\n梯度下降后：", forward(x), "\ncost：", cost(forward(x), y))


if __name__ == '__main__':
    setup()

```
&emsp;&emsp;结果如下：可以看见，结果确实是在逐步想着目标结果靠近，cost值不断在减小。证明我们的算法是正确的。
```text
梯度下降前： [[0.90024951 0.94267582]
 [0.80218389 0.88079708]]
梯度下降后： [[0.87638772 0.93574933]
 [0.74070904 0.87317416]]
cost： 1.4556414717601052
梯度下降前： [[0.87638772 0.93574933]
 [0.74070904 0.87317416]]
梯度下降后： [[0.84537106 0.92722992]
 [0.66043307 0.86435062]]
cost： 1.3382380273209387
梯度下降前： [[0.84537106 0.92722992]
 [0.66043307 0.86435062]]
梯度下降后： [[0.80943371 0.91658752]
 [0.56909364 0.85403973]]
cost： 1.2216201634346886
梯度下降前： [[0.80943371 0.91658752]
 [0.56909364 0.85403973]]
梯度下降后： [[0.77530479 0.90307287]
 [0.48379806 0.84187495]]
cost： 1.1250926413642042
梯度下降前： [[0.77530479 0.90307287]
 [0.48379806 0.84187495]]
梯度下降后： [[0.74994151 0.88562481]
 [0.41750738 0.8273968 ]]
cost： 1.050964830757349
梯度下降前： [[0.74994151 0.88562481]
 [0.41750738 0.8273968 ]]
梯度下降后： [[0.73518018 0.86276177]
 [0.37075203 0.81005788]]
cost： 0.9880224900744513
梯度下降前： [[0.73518018 0.86276177]
 [0.37075203 0.81005788]]
梯度下降后： [[0.7288795  0.83251337]
 [0.33814879 0.78928408]]
cost： 0.9253306342190869
梯度下降前： [[0.7288795  0.83251337]
 [0.33814879 0.78928408]]
梯度下降后： [[0.72817698 0.7925729 ]
 [0.31464068 0.76467358]]
cost： 0.8564368364354394
梯度下降前： [[0.72817698 0.7925729 ]
 [0.31464068 0.76467358]]
梯度下降后： [[0.73084978 0.74107485]
 [0.29686131 0.73646017]]
cost： 0.7792136510576879
梯度下降前： [[0.73084978 0.74107485]
 [0.29686131 0.73646017]]
梯度下降后： [[0.73542993 0.67843692]
 [0.28276129 0.70627592]]
cost： 0.6965017597370333
梯度下降前： [[0.73542993 0.67843692]
 [0.28276129 0.70627592]]
梯度下降后： [[0.74100699 0.60952933]
 [0.27110861 0.67770711]]
cost： 0.6159759746541653
梯度下降前： [[0.74100699 0.60952933]
 [0.27110861 0.67770711]]
梯度下降后： [[0.7470327  0.54300877]
 [0.2611523  0.65537568]]
cost： 0.5458174231304158
梯度下降前： [[0.7470327  0.54300877]
 [0.2611523  0.65537568]]
梯度下降后： [[0.75318337 0.48629069]
 [0.25242337 0.64233397]]
cost： 0.48903961977358096
梯度下降前： [[0.75318337 0.48629069]
 [0.25242337 0.64233397]]
梯度下降后： [[0.75927196 0.44162035]
 [0.24462032 0.63846146]]
cost： 0.4435277424325401
梯度下降前： [[0.75927196 0.44162035]
 [0.24462032 0.63846146]]
梯度下降后： [[0.76519387 0.40729807]
 [0.23754304 0.64153151]]
cost： 0.4059519984224861
梯度下降前： [[0.76519387 0.40729807]
 [0.23754304 0.64153151]]
梯度下降后： [[0.77089406 0.38056044]
 [0.23105412 0.64898998]]
cost： 0.3739098246421919
梯度下降前： [[0.77089406 0.38056044]
 [0.23105412 0.64898998]]
梯度下降后： [[0.77634715 0.35906729]
 [0.22505587 0.65883242]]
cost： 0.3459953751213052
梯度下降前： [[0.77634715 0.35906729]
 [0.22505587 0.65883242]]
梯度下降后： [[0.78154526 0.34118504]
 [0.21947641 0.66972652]]
cost： 0.3213801718742878
梯度下降前： [[0.78154526 0.34118504]
 [0.21947641 0.66972652]]
梯度下降后： [[0.78649079 0.32585178]
 [0.21426107 0.68086606]]
cost： 0.29951983756020983
......
梯度下降前： [[0.97352909 0.02666433]
 [0.02647091 0.97333567]]
梯度下降后： [[0.97354315 0.02664997]
 [0.02645685 0.97335003]]
cost： 0.002820371366327034
梯度下降前： [[0.97354315 0.02664997]
 [0.02645685 0.97335003]]
梯度下降后： [[0.97355719 0.02663563]
 [0.02644281 0.97336437]]
cost： 0.002817357738697952
梯度下降前： [[0.97355719 0.02663563]
 [0.02644281 0.97336437]]
梯度下降后： [[0.97357121 0.02662131]
 [0.02642879 0.97337869]]
cost： 0.002814350458880144
梯度下降前： [[0.97357121 0.02662131]
 [0.02642879 0.97337869]]
梯度下降后： [[0.9735852  0.02660701]
 [0.0264148  0.97339299]]
cost： 0.0028113495069739336
梯度下降前： [[0.9735852  0.02660701]
 [0.0264148  0.97339299]]
梯度下降后： [[0.97359917 0.02659274]
 [0.02640083 0.97340726]]
cost： 0.002808354863162452
梯度下降前： [[0.97359917 0.02659274]
 [0.02640083 0.97340726]]
梯度下降后： [[0.97361312 0.02657849]
 [0.02638688 0.97342151]]
cost： 0.0028053665077110495
梯度下降前： [[0.97361312 0.02657849]
 [0.02638688 0.97342151]]
梯度下降后： [[0.97362705 0.02656426]
 [0.02637295 0.97343574]]
cost： 0.002802384420967047
梯度下降前： [[0.97362705 0.02656426]
 [0.02637295 0.97343574]]
梯度下降后： [[0.97364096 0.02655005]
 [0.02635904 0.97344995]]
cost： 0.002799408583359122
梯度下降前： [[0.97364096 0.02655005]
 [0.02635904 0.97344995]]
梯度下降后： [[0.97365484 0.02653587]
 [0.02634516 0.97346413]]
cost： 0.002796438975397068
梯度下降前： [[0.97365484 0.02653587]
 [0.02634516 0.97346413]]
梯度下降后： [[0.97366871 0.02652171]
 [0.02633129 0.97347829]]
cost： 0.002793475577671274
梯度下降前： [[0.97366871 0.02652171]
 [0.02633129 0.97347829]]
梯度下降后： [[0.97368255 0.02650757]
 [0.02631745 0.97349243]]
cost： 0.0027905183708523346
梯度下降前： [[0.97368255 0.02650757]
 [0.02631745 0.97349243]]
梯度下降后： [[0.97369637 0.02649345]
 [0.02630363 0.97350655]]
cost： 0.002787567335690624
梯度下降前： [[0.97369637 0.02649345]
 [0.02630363 0.97350655]]
梯度下降后： [[0.97371017 0.02647935]
 [0.02628983 0.97352065]]
cost： 0.0027846224530159356

```