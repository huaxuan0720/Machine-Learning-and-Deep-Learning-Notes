---
title: 反向传播算法（一）之反向传播入门
date: 2019-05-12 21:18:21
tags: [机器学习 ,反向传播算法]
categories: 机器学习
toc: true
thumbnail: gallery/MachineLearning.jpg
---

##### 一、反向传播算法
&emsp;&emsp;近年来，深度学习的快速发展带来了一系列喜人的成果，不管是在图像领域还是在NLP领域，深度学习都显示了其极其强大的能力。而深度学习之所以可以进行大规模的参数运算，反向传播算法功不可没，可以说，没有反向传播算法，深度学习就不可能得以快速发展，因此在此之前，有必要了解一下反向传播算法的具体原理和公式推导。请注意：这里的所有推导过程都只是针对当前设置的参数信息，并不具有一般性，但是所有的推导过程可以推导到一般的运算，因此以下给出的并不是反向传播算法的严格证明，但是可以很好的帮助理解反向传播算法。  
<!--more-->  

##### 二、梯度下降
&emsp;&emsp;首先反向传播算法的核心思路就是梯度下降，那么我们必须要明白什么是梯度，从几何上理解，一个函数（此处默认该函数处处可导）的图像会在其空间内呈现出一个曲面（曲线）。以 $f(x) = x^2 + y ^2$ 为例，该函数会在三维空间（x, y, z）中形成一个曲面，其中，x, y可看作相互独立的两个变量，那么我们分别对x,y求偏导数，会有 $\frac{\partial f}{\partial x} = 2x$，$\frac{\partial f}{\partial y} = 2y$，因此，该函数的梯度可以表示为 $(2x, 2y)$，如在坐标（1， 3）处的梯度，代入公式可以得到（2， 6）。该数值表示在x, y构成的平面上，我们首先所处的位置是（1， 3）点，该处的函数值是10。如果我需要以最快的速度增大函数值，那么我需要根据 *向量（2，6）* 的方向前进。  
&emsp;&emsp;因此梯度在几何上的直观理解是一个表明方向的向量，只有朝着这个向量所指示的方向前进，函数值才会增加的最快。可以这么说（不算很严谨），梯度所在的空间是所有的自变量构成的空间，并指示着自变量需要变化的方向。  
&emsp;&emsp;由于梯度指示的是函数值增大最快的方向，那么我们朝着相反的方向前进，函数值也必定会下降最快（所以我们在公式中是减去梯度，而不是加上梯度），这就是梯度下降算法的核心。由于梯度值只在一个很小的范围内近似保持不变，所以我们需要进行迭代，并且需要用一个步长变量来控制下降的幅度，这个步长变量就是我们经常谈到的学习率。

##### 三、单层全连接层以及单个输出，不使用激活函数  
&emsp;&emsp;在所有的矩阵相乘的情况中，输入输出之间只有一个全连接层并且该全连接层的输出仅仅是一个常数（或者说是一个1x1大小的矩阵），同时并不使用非线性激活函数的情况是最简单的，因此，首先可以考虑这种最简单的情况。

&emsp;&emsp;我们这里假设输入$x$是一个长度为3的向量，按照Tensor Flow所设定的数据格式，我们这里设定该输入向量为行向量，即$x = \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix}$，输出为一个回归值 $\hat{y}$，目标回归值为$y$，全连接层权值记为 $\omega$，（$\omega$是一个矩阵，大小为3x1），偏置项记为$b$，（$b$也可以视作一个矩阵，大小为1x1，也可以看作是一个数值，因为这里不关心$b$的维度信息，因此不做严格的区分。）。当我们定义了以上的相关参数之后，就可以进行如下的运算：
$$
x * \omega + b = \hat{y} \tag{1}
$$
&emsp;&emsp;即：
$$
\begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix} * \begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \end{bmatrix} + b = \hat{y} \tag{2}
$$

&emsp;&emsp;这里取损失cost的计算方式为差值的平方，即 $C = cost(\hat{y}, y) = (\hat{y} - y)^2$，很显然，我们对 $\hat{y}$ 计算偏导数（导数）可得到： $\frac{\partial C}{\partial \hat{y}} = 2 (\hat{y} - y)$。

&emsp;&emsp;将上面的(2)式展开，可以得到下面的多项式：
$$
\omega_1 x_1 + \omega_2 x_2 + \omega_3 x_3 + b = \hat{y} \tag{3}
$$

&emsp;&emsp;因为我们关心的是 $\omega_1$，$\omega_2$，$\omega_3$ 以及 $b$ 的更新梯度，因此我们需要对这四个变量求偏导数。根据 $\hat{y}$ 的计算公式，我们不难看出这四个变量的偏导数计算公式如下：
$$
\frac{\partial \hat{y}}{\partial \omega_1} = x_1，\frac{\partial \hat{y}}{\partial \omega_2} = x_2，\frac{\partial \hat{y}}{\partial \omega_3} = x_3，\frac{\partial \hat{y}}{\partial b} = 1 \tag{4}
$$
&emsp;&emsp;本质上，我们需要将$C$的数值降低到全局最小值（或者局部最小值），因此我们需要根据 $\frac{\partial C}{\partial \omega_1}$，$\frac{\partial C}{\partial \omega_2}$，$\frac{\partial C}{\partial \omega_3}$ 和 $\frac{\partial C}{\partial b}$ 这四个参数的梯度信息来更新相关参数的数值。根据求导公式的链式法则（chain rule），我们有：
$$
\frac{\partial C}{\partial \omega_1} = \frac{\partial C}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \omega_1} = 2 (\hat{y} - y) \cdot x_1 \tag{5}
$$
$$
\frac{\partial C}{\partial \omega_2} = \frac{\partial C}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \omega_2} = 2 (\hat{y} - y) \cdot x_2 \tag{6}
$$
$$
\frac{\partial C}{\partial \omega_3} = \frac{\partial C}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial \omega_3} = 2 (\hat{y} - y) \cdot x_3 \tag{7}
$$
$$
\frac{\partial C}{\partial b} = \frac{\partial C}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial b} = 2 (\hat{y} - y) \cdot 1 \tag{8}
$$

&emsp;&emsp;上式(5)~(8)就是我们计算得到的梯度信息，根据梯度信息，我们就可以更新相关的参数了，以下公式中的 $\alpha$ 表示的是学习率，为人为设置的一个超参数。
$$
\omega_1 := \omega_1 - \alpha \cdot \frac{\partial C}{\partial \omega_1} = \omega_1 - \alpha \cdot (2 (\hat{y} - y) \cdot x_1) \tag{9}
$$
$$
\omega_2 := \omega_2 - \alpha \cdot \frac{\partial C}{\partial \omega_2} = \omega_2 - \alpha \cdot (2 (\hat{y} - y) \cdot x_2) \tag{10}
$$
$$
\omega_3 := \omega_3 - \alpha \cdot \frac{\partial C}{\partial \omega_3} = \omega_3 - \alpha \cdot (2 (\hat{y} - y) \cdot x_3) \tag{11}
$$
$$
b := b - \alpha \cdot \frac{\partial C}{\partial b} = b - \alpha \cdot (2 (\hat{y} - y)) \tag{12}
$$

&emsp;&emsp;以上的公式就已经可以用来进行反向传播，或者说梯度下降了，但是实际上，在代码编写的时候，直接使用上面的公式会显得非常繁琐，因此，我们常常使用上面公式的向量化表达，这样可以使代码编写简洁高效，并且由于numpy等python包对向量和矩阵运算进行了很大程度的优化，因此运算速度也比直接使用上述公式要快。

&emsp;&emsp;我们将每个变量的梯度按照次序排好，放入一个矩阵中，如下：
$$
\begin{bmatrix} \frac{\partial C}{\partial \omega_1} \\ \frac{\partial C}{\partial \omega_2} \\ \frac{\partial C}{\partial \omega_3} \end{bmatrix} = \begin{bmatrix} 2(\hat{y} - y) \cdot x_1 \\ 2(\hat{y} - y) \cdot x_2 \\ 2(\hat{y} - y) \cdot x_3 \\ \end{bmatrix} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \begin{bmatrix} 2(\hat{y} - y) \end{bmatrix} = x^T \begin{bmatrix} 2(\hat{y} - y) \end{bmatrix}
$$
&emsp;&emsp;化简之后，$\omega$ 权值梯度更新的公式如下：
$$
\frac{\partial C}{\partial \omega} = x^T \begin{bmatrix} 2(\hat{y} - y) \end{bmatrix}
$$
$$
\omega := \omega - \alpha * x^T \begin{bmatrix} 2(\hat{y} - y) \end{bmatrix}
$$


##### 四、代码  
&emsp;&emsp;
```python
import numpy as np

param = {}
nodes = {}

learning_rate = 0.001


def forward(x):
    nodes["matmul"] = np.matmul(x, param["w"])
    nodes['add'] = nodes['matmul'] + param["b"]
    return nodes['add']


def cost(y_pred, y):
    return np.sum((y_pred - y) ** 2)


def cost_gradient(y_pred, y):
    return 2 * (y_pred - y)


def backward(x, y_pred, y):
    param['w'] -= learning_rate * np.matmul(np.transpose(x), cost_gradient(y_pred, y))
    param['b'] -= learning_rate * cost_gradient(y_pred, y)


def setup():
    param["w"] = np.random.random([3, 1])
    param["b"] = np.random.random([1, 1])

    x = np.array([[1., 2., 3.]])
    y = np.array([[1]])

    for i in range(200):
        y_pred = forward(x)
        backward(x, y_pred, y)
        print("预测结果：", y_pred, " 梯度下降之后：", forward(x), " 真实回归值：", y, " Loss：", cost(y_pred, y))
    pass


if __name__ == '__main__':
    setup()

```
&emsp;&emsp;结果如下：
```text
预测结果： [[3.35507743]]  梯度下降之后： [[3.28442511]]  真实回归值： [[1]]  Loss： 5.546389698793334
预测结果： [[3.28442511]]  梯度下降之后： [[3.21589235]]  真实回归值： [[1]]  Loss： 5.218598067594647
预测结果： [[3.21589235]]  梯度下降之后： [[3.14941558]]  真实回归值： [[1]]  Loss： 4.910178921799803
预测结果： [[3.14941558]]  梯度下降之后： [[3.08493312]]  真实回归值： [[1]]  Loss： 4.619987347521437
预测结果： [[3.08493312]]  梯度下降之后： [[3.02238512]]  真实回归值： [[1]]  Loss： 4.346946095282921
预测结果： [[3.02238512]]  梯度下降之后： [[2.96171357]]  真实回归值： [[1]]  Loss： 4.090041581051698
预测结果： [[2.96171357]]  梯度下降之后： [[2.90286216]]  真实回归值： [[1]]  Loss： 3.8483201236115456
......
预测结果： [[2.16884003]]  梯度下降之后： [[2.13377483]]  真实回归值： [[1]]  Loss： 1.366187026290616
预测结果： [[2.13377483]]  梯度下降之后： [[2.09976159]]  真实回归值： [[1]]  Loss： 1.2854453730368411
预测结果： [[2.09976159]]  梯度下降之后： [[2.06676874]]  真实回归值： [[1]]  Loss： 1.2094755514903637
预测结果： [[2.06676874]]  梯度下降之后： [[2.03476568]]  真实回归值： [[1]]  Loss： 1.137995546397283
预测结果： [[2.03476568]]  梯度下降之后： [[2.00372271]]  真实回归值： [[1]]  Loss： 1.0707400096052042
预测结果： [[2.00372271]]  梯度下降之后： [[1.97361103]]  真实回归值： [[1]]  Loss： 1.0074592750375366
预测结果： [[1.97361103]]  梯度下降之后： [[1.9444027]]  真实回归值： [[1]]  Loss： 0.9479184318828179
预测结果： [[1.9444027]]  梯度下降之后： [[1.91607062]]  真实回归值： [[1]]  Loss： 0.8918964525585433
预测结果： [[1.91607062]]  梯度下降之后： [[1.8885885]]  真实回归值： [[1]]  Loss： 0.8391853722123339
......
预测结果： [[1.6356086]]  梯度下降之后： [[1.61654034]]  真实回归值： [[1]]  Loss： 0.40399829055941733
预测结果： [[1.61654034]]  梯度下降之后： [[1.59804413]]  真实回归值： [[1]]  Loss： 0.38012199158735577
预测结果： [[1.59804413]]  梯度下降之后： [[1.58010281]]  真实回归值： [[1]]  Loss： 0.3576567818845428
预测结果： [[1.58010281]]  梯度下降之后： [[1.56269972]]  真实回归值： [[1]]  Loss： 0.3365192660751663
预测结果： [[1.56269972]]  梯度下降之后： [[1.54581873]]  真实回归值： [[1]]  Loss： 0.316630977450124
预测结果： [[1.54581873]]  梯度下降之后： [[1.52944417]]  真实回归值： [[1]]  Loss： 0.29791808668282155
预测结果： [[1.52944417]]  梯度下降之后： [[1.51356084]]  真实回归值： [[1]]  Loss： 0.2803111277598668
预测结果： [[1.51356084]]  梯度下降之后： [[1.49815402]]  真实回归值： [[1]]  Loss： 0.2637447401092591
......
预测结果： [[1.00722162]]  梯度下降之后： [[1.00700497]]  真实回归值： [[1]]  Loss： 5.215181191515595e-05
预测结果： [[1.00700497]]  梯度下降之后： [[1.00679482]]  真实回归值： [[1]]  Loss： 4.906963983096906e-05
预测结果： [[1.00679482]]  梯度下降之后： [[1.00659098]]  真实回归值： [[1]]  Loss： 4.616962411696138e-05
预测结果： [[1.00659098]]  梯度下降之后： [[1.00639325]]  真实回归值： [[1]]  Loss： 4.344099933164832e-05
预测结果： [[1.00639325]]  梯度下降之后： [[1.00620145]]  真实回归值： [[1]]  Loss： 4.087363627114841e-05
预测结果： [[1.00620145]]  梯度下降之后： [[1.00601541]]  真实回归值： [[1]]  Loss： 3.84580043675206e-05
预测结果： [[1.00601541]]  梯度下降之后： [[1.00583495]]  真实回归值： [[1]]  Loss： 3.618513630940355e-05
预测结果： [[1.00583495]]  梯度下降之后： [[1.0056599]]  真实回归值： [[1]]  Loss： 3.4046594753515e-05
预测结果： [[1.0056599]]  梯度下降之后： [[1.0054901]]  真实回归值： [[1]]  Loss： 3.203444100358307e-05
预测结果： [[1.0054901]]  梯度下降之后： [[1.0053254]]  真实回归值： [[1]]  Loss： 3.0141205540271894e-05
```
&emsp;&emsp;可以发现，经过梯度下降之后，预测的回归值逐渐接近真实的回归值，loss也一直在不断降低，证明我们的算法是正确的。

