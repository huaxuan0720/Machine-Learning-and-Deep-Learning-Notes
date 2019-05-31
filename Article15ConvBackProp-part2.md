---
title: 步长stride为s的二维卷积方法的反向传播算法
date: 2019-05-24 12:32:31
tags: [卷积, 反向传播, 深度学习]
category: 深度学习
toc: true
thumbnail: gallery/DeepLearning.jpg
---

##### 前言  

&emsp;&emsp;在之前讨论了步长stride为1的卷积方式的反向传播，但是很多时候，使用的卷积步长会大于1，这个情况下的卷积方式的反向传播和步长为1的情况稍稍有些区别，不过区别并没有想象中那么大，因此下面就对步长stride大于1的情况进行简单的阐述。请注意：这里的所有推导过程都只是针对当前设置的参数信息，并不具有一般性，但是所有的推导过程可以推导到一般的运算，因此以下给出的并不是反向传播算法的严格证明，不涉及十分复杂的公式推导，争取可以以一种简单的方式来理解卷积的反向传播。希望可以很好的帮助理解反向传播算法。   

&emsp;&emsp;需要注意的是，在本文中，所有的正向传播过程中，卷积的步长stride均固定为2。  

<!--more-->

##### 一，参数设置   

&emsp;&emsp;这里我们设置我们的数据矩阵（记作$x$）大小为5x5，卷积核（记作$k$）大小为3x3，由于步长是2，因此，卷积之后获得的结果是一个2x2大小的数据矩阵（不妨我们记作$u$）。偏置项我们记为$b$，将和卷积之后的矩阵进行相加。  

&emsp;&emsp;我们的参数汇总如下：  

| 参数          | 设置                    |
| ------------- | ----------------------- |
| 输入矩阵$x$   | 一个二维矩阵，大小为5x5 |
| 输入卷积核$k$ | 一个二维矩阵，大小为3x3 |
| 步长$stride$  | 设置为2                 |
| padding       | VALID                   |
| 偏置项$b$     | 一个浮点数              |

&emsp;&emsp;和前面一样，我们定义卷积操作的符号为$conv$，我们可以将卷积表示为（需要注意的是这里步长选取为2）：  

$$
x \; conv \; k + b = u
$$

&emsp;&emsp;展开之后，我们可以得到：  

$$
\begin{bmatrix}
x_{1, 1} & x_{1, 2} & x_{1, 3} &x_{1, 4} &x_{1, 5} \\
x_{2, 1} & x_{2, 2} & x_{2, 3} &x_{2, 4} &x_{2, 5} \\
x_{3, 1} & x_{3, 2} & x_{3, 3} &x_{3, 4} &x_{3, 5} \\
x_{4, 1} & x_{4, 2} & x_{4, 3} &x_{4, 4} &x_{4, 5} \\
x_{5, 1} & x_{5, 2} & x_{5, 3} &x_{5, 4} &x_{5, 5} \\
\end{bmatrix} \; conv \;
\begin{bmatrix}
k_{1, 1} & k_{1, 2} & k_{1, 3}\\
k_{2, 1} & k_{2, 2} & k_{2, 3}\\
k_{3, 1} & k_{3, 2} & k_{3, 3}\\
\end{bmatrix} + b = 
\begin{bmatrix}
u_{1, 1} & u_{1, 2} \\
u_{2, 1} & u_{2, 2} \\
\end{bmatrix}
$$

&emsp;&emsp;将矩阵$u$进一步展开，我们有：  

$$
\begin{bmatrix}
u_{1, 1} & u_{1, 2} \\
u_{2, 1} & u_{2, 2} \\
\end{bmatrix} = \\ 
\begin{bmatrix}
\begin{matrix}
x_{1, 1}k_{1, 1} + x_{1, 2}k_{1, 2} +x_{1, 3}k_{1, 3} + \\
x_{2, 1}k_{2, 1} + x_{2, 2}k_{2, 2} +x_{2, 3}k_{2, 3} + \\
x_{3, 1}k_{3, 1} + x_{3, 2}k_{3, 2} +x_{3, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{1, 3}k_{1, 1} + x_{1, 4}k_{1, 2} +x_{1, 5}k_{1, 3} + \\
x_{2, 3}k_{2, 1} + x_{2, 4}k_{2, 2} +x_{2, 5}k_{2, 3} + \\
x_{3, 3}k_{3, 1} + x_{3, 4}k_{3, 2} +x_{3, 5}k_{3, 3} + b \\
\end{matrix} \\ \\
\begin{matrix}
x_{3, 1}k_{1, 1} + x_{3, 2}k_{1, 2} +x_{3, 3}k_{1, 3} + \\
x_{4, 1}k_{2, 1} + x_{4, 2}k_{2, 2} +x_{4, 3}k_{2, 3} + \\
x_{5, 1}k_{3, 1} + x_{5, 2}k_{3, 2} +x_{5, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{3, 3}k_{1, 1} + x_{3, 4}k_{1, 2} +x_{3, 5}k_{1, 3} + \\
x_{4, 3}k_{2, 1} + x_{4, 4}k_{2, 2} +x_{4, 5}k_{2, 3} + \\
x_{5, 3}k_{3, 1} + x_{5, 4}k_{3, 2} +x_{5, 5}k_{3, 3} + b \\
\end{matrix} \\
\end{bmatrix}
$$

##### 二、误差传递   

&emsp;&emsp;步长为2的二维卷积已经在上面的式子中被完整的表示出来了，因此，下一步就是需要对误差进行传递，和前面步长为1的情况一样，我们可以将上面的结果保存在一张表格中，每一列表示的是一个特定的输出 $\partial u_{i, j}$，每一行表示的是一个特定的输入值$\partial x_{p, k}$，行与列相交的地方表示的就是二者相除的结果，表示的是输出对于输入的偏导数，即$\frac{\partial u_{i, j}}{\partial x_{p, k}}$。于是，表格如下：  

|      |$\partial u_{1, 1}$|$\partial u_{1, 2}$|$\partial u_{2, 1}$|$\partial u_{2, 2}$|$\frac{\partial L}{\partial x_{i, j}}$|
| ---- | :--: | :--: | :--: | :--: | :--- |
| $\partial x_{1, 1}$ |     $k_{1, 1}$      |          0          |          0          |          0          |$\frac{\partial L}{\partial x_{1, 1}} = \delta_{1, 1} k_{1, 1}$|
| $\partial x_{1, 2}$ |     $k_{1, 2}$      |          0          |          0          |          0          |$\frac{\partial L}{\partial x_{1, 2}} = \delta_{1, 1} k_{1, 2}$|
| $\partial x_{1, 3}$ |     $k_{1, 3}$      |          $k_{1, 1}$          |          0          |          0          |$\frac{\partial L}{\partial x_{1, 3}} = \delta_{1, 1} k_{1, 3} + \delta_{1, 2}k_{1, 1}$|
| $\partial x_{1, 4}$ |     0     |          $k_{1, 2}$          |          0          |          0          |$\frac{\partial L}{\partial x_{1, 4}} = \delta_{1, 2}k_{1, 2}$|
| $\partial x_{1, 5}$ |     0     |          $k_{1, 3}$          |          0          |          0          |$\frac{\partial L}{\partial x_{1, 5}} = \delta_{1, 2}k_{1, 3}$|
| $\partial x_{2, 1}$ |     $k_{2, 1}$      |          0          |          0          |          0          |$\frac{\partial L}{\partial x_{2, 1}} = \delta_{1, 1} k_{2, 1}$|
| $\partial x_{2, 2}$ |     $k_{2, 2}$      |          0          |          0          |          0          |$\frac{\partial L}{\partial x_{2, 2}} = \delta_{1, 1} k_{2, 2}$|
| $\partial x_{2, 3}$ |     $k_{2, 3}$      |          $k_{2, 1}$          |          0          |          0          |$\frac{\partial L}{\partial x_{2, 3}} = \delta_{1, 1} k_{1, 3} + \delta_{1, 2}k_{2, 1}$|
| $\partial x_{2, 4}$ |     0     |          $k_{2, 2}$          |          0          |          0          |$\frac{\partial L}{\partial x_{2, 4}} = \delta_{1, 2}k_{2, 2}$|
| $\partial x_{2, 5}$ |     0     |          $k_{2, 3}$          |          0          |          0          |$\frac{\partial L}{\partial x_{2, 5}} = \delta_{1, 2}k_{2, 3}$|
| $\partial x_{3, 1}$ |     $k_{3, 1}$      |          0          |          $k_{1, 1}$          |          0          |$\frac{\partial L}{\partial x_{3, 1}} = \delta_{1, 1}k_{3, 1} + \delta_{2, 1}k_{1, 1}$|
| $\partial x_{3, 2}$ |     $k_{3, 2}$     |          0          |          $k_{1, 2}$          |          0          |$\frac{\partial L}{\partial x_{3, 2}} = \delta_{1, 1}k_{3, 2} + \delta_{2, 1}k_{1, 2}$|
| $\partial x_{3, 3}$ |     $k_{3, 3}$      |          $k_{3, 1}$          |          $k_{1, 3}$          |          $k_{1, 1}$          |$\frac{\partial L}{\partial x_{3, 3}} = \delta_{1, 1}k_{3, 3} + \delta_{1, 2}k_{3, 1} + \delta_{2, 1}k_{1, 3} + \delta_{2, 2}k_{1, 1}$|
| $\partial x_{3, 4}$ |     0     |          $k_{3, 2}$          |          0          |          $k_{1, 2}$          |$\frac{\partial L}{\partial x_{3, 4}} = \delta_{1, 2}k_{3, 2} + \delta_{2, 2}k_{1, 2}$|
| $\partial x_{3, 5}$ |     0     |          $k_{3, 3}$          |          0          |          $k_{1, 3}$          |$\frac{\partial L}{\partial x_{3, 5}} = \delta_{1, 2}k_{3, 3} + \delta_{2, 2}k_{1, 3}$|
| $\partial x_{4, 1}$ |     0     |          0          |          $k_{2, 1}$          |          0          |$\frac{\partial L}{\partial x_{4, 1}} = \delta_{2, 1}k_{2, 1}$|
| $\partial x_{4, 2}$ |     0     |          0          |          $k_{2, 2}$          |          0          |$\frac{\partial L}{\partial x_{4, 2}} = \delta_{2, 1}k_{2, 2}$|
| $\partial x_{4, 3}$ |     0     |          0          |          $k_{2, 3}$          |          $k_{2, 1}$          |$\frac{\partial L}{\partial x_{4, 3}} = \delta_{2, 1}k_{2, 3} +  \delta_{2, 2}k_{2, 1}$|
| $\partial x_{4, 4}$ |     0     |          0          |          0          |          $k_{2, 2}$          |$\frac{\partial L}{\partial x_{4, 4}} = \delta_{2, 2}k_{2, 2}$|
| $\partial x_{4, 5}$ |     0     |          0          |          0          |          $k_{2, 3}$          |$\frac{\partial L}{\partial x_{4, 5}} = \delta_{2, 2}k_{2, 3}$|
| $\partial x_{5, 1}$ |     0     |          0          |          $k_{3, 1}$          |          0          |$\frac{\partial L}{\partial x_{5, 1}} = \delta_{2, 1}k_{3, 1}$|
| $\partial x_{5, 2}$ |     0     |          0          |          $k_{3, 2}$          |          0          |$\frac{\partial L}{\partial x_{5, 2}} = \delta_{2, 1}k_{3, 2}$|
| $\partial x_{5, 3}$ |     0      |          0          |          $k_{3, 3}$          |          $k_{3, 1}$          |$\frac{\partial L}{\partial x_{5, 3}} = \delta_{2, 1}k_{3, 3} + \delta_{2, 2}k_{3, 1}$|
| $\partial x_{5, 4}$ |     0     |          0          |          0          |          $k_{3, 2}$          |$\frac{\partial L}{\partial x_{5, 4}} = \delta_{2, 2}k_{3, 2}$|
| $\partial x_{5, 5}$ |     0     |          0          |          0          |          $k_{3, 3}$          |$\frac{\partial L}{\partial x_{5, 5}} = \delta_{2, 2}k_{3, 3}$|

&emsp;&emsp;可以看出，数据依然都是很规律的进行着重复。  

&emsp;&emsp;我们假设后面传递过来的误差是 $\delta$ ，即：  

$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} \\
\delta_{2, 1} & \delta_{2, 2} \\
\end{bmatrix}
$$

&emsp;&emsp;其中，$\delta_{i, j} = \frac{\partial L}{\partial u_{i, j}}$，误差分别对应于每一个输出项。这里的$L$表示的是最后的Loss损失。我们的目的就是希望这个损失尽可能小。那么，根据求导的链式法则，我们有：  

&emsp;&emsp;根据求偏导数的链式法则，我们可以有：  

$$
\frac{\partial L}{\partial x_{i, j}} = \sum_{p = 1} \sum_{k = 1} \frac{\partial L}{\partial u_{p, k}} \cdot \frac{\partial u_{p, k}}{\partial x_{i, j}} = \sum_{p = 1} \sum_{k = 1} \delta_{p, k} \cdot \frac{\partial u_{p, k}}{\partial x_{i, j}}
$$

&emsp;&emsp;我们以$\frac{\partial L}{\partial x_{3, 3}}$为例，我们有：  

$$
\begin{aligned}
\frac{\partial L}{\partial x_{3, 3}} 
&= \sum_{p = 1} \sum_{k = 1} \frac{\partial L}{\partial u_{p, k}} \cdot \frac{\partial u_{p, k}}{\partial x_{3, 3}} \\
&= \sum_{p = 1} \sum_{k = 1} \delta_{p, k} \cdot \frac{\partial u_{p, k}}{\partial x_{3, 3}} \\
&= \delta_{1, 1}\frac{\partial u_{1, 1}}{\partial x_{3, 3}} + \delta_{1, 2}\frac{\partial u_{1, 2}}{\partial x_{3, 3}} + \delta_{2, 1}\frac{\partial u_{2, 1}}{\partial x_{3, 3}} + \delta_{2, 2}\frac{\partial u_{2, 2}}{\partial x_{3, 3}} \\
&= \delta_{1, 1}k_{3, 3} + \delta_{1, 2}k_{3, 1} + \delta_{2, 1}k_{1, 3} + \delta_{2, 2}k_{1, 1}
\end{aligned}
$$

&emsp;&emsp; 类似地，我们可以计算出所有的输入矩阵中的元素所对应的偏导数信息，所有的偏导数计算结果均在上表中列出。  

&emsp;&emsp;和前面步长stride为1的卷积方式的误差传递类似，我们需要对传递来的误差矩阵和卷积核进行一定的处理，然后再进行卷积，得到应该传递到下一层的网络结构中，所以我们需要的解决问题的问题有三个，即：1.误差矩阵如何处理，2.卷积核如何处理，3.如何进行卷积。  

&emsp;&emsp;同样，我们将$\frac{\partial L}{\partial x_{3, 3}}$单独拿出来进行考察，如果需要用到全部的卷积核的元素的话，并不能和传递来的误差矩阵相匹配，为了使得两者可以再维度上相匹配，我们再误差矩阵中添加若干0，和步长stride为1的卷积反向传播一样，我们也将卷积核进行180°翻转，于是，我们可以得到：  

$$
\frac{\partial L}{\partial x_{3, 3}} = \begin{bmatrix}
\delta_{1, 1} & 0 & \delta_{1, 2} \\
0 & 0 & 0 \\
\delta_{2, 1} & 0 & \delta_{2, 2} \\
\end{bmatrix}
\; conv \;\begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$

&emsp;&emsp;由于padding策略一直默认为是VALID，而且上面的两个矩阵形状相同，所以此时的步长stride参数不会影响到最终的结果。  

&emsp;&emsp;如果按照我们之前的策略，再在添加0之后的误差矩阵外面填补上合适数目的0的话，有：  

$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{1, 1} & 0 & \delta_{1, 2} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{2, 1} & 0 & \delta_{2, 2} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
\; conv \;\begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$

&emsp;&emsp;同样，上面的卷积过程步长stride参数为1。  

&emsp;&emsp;不妨将上面的卷积的结果记为$conv1$，然后，我们将$\frac{\partial L}{\partial x_{i, j}}$按照对应的顺序进行排列，我们将结果记作$conv2$，即：  

$$
conv2 = \begin{bmatrix}
\frac{\partial L}{\partial x_{1, 1}} & \frac{\partial L}{\partial x_{1, 2}} & \frac{\partial L}{\partial x_{1, 3}}& \frac{\partial L}{\partial x_{1, 4}} & \frac{\partial L}{\partial x_{1, 5}} \\
\frac{\partial L}{\partial x_{2, 1}} & \frac{\partial L}{\partial x_{2, 2}} & \frac{\partial L}{\partial x_{2, 3}}& \frac{\partial L}{\partial x_{2, 4}} & \frac{\partial L}{\partial x_{2, 5}} \\
\frac{\partial L}{\partial x_{3, 1}} & \frac{\partial L}{\partial x_{3, 2}} & \frac{\partial L}{\partial x_{3, 3}}& \frac{\partial L}{\partial x_{3, 4}} & \frac{\partial L}{\partial x_{3, 5}} \\
\frac{\partial L}{\partial x_{4, 1}} & \frac{\partial L}{\partial x_{4, 2}} & \frac{\partial L}{\partial x_{4, 3}}& \frac{\partial L}{\partial x_{4, 4}} & \frac{\partial L}{\partial x_{4, 5}} \\
\frac{\partial L}{\partial x_{5, 1}} & \frac{\partial L}{\partial x_{5, 2}} & \frac{\partial L}{\partial x_{5, 3}}& \frac{\partial L}{\partial x_{5, 4}} & \frac{\partial L}{\partial x_{5, 5}} \\
\end{bmatrix}
$$

&emsp;&emsp;经过计算，我们发现$conv1$和$conv2$正好相等。即：  

$$
\begin{bmatrix}
\frac{\partial L}{\partial x_{1, 1}} & \frac{\partial L}{\partial x_{1, 2}} & \frac{\partial L}{\partial x_{1, 3}}& \frac{\partial L}{\partial x_{1, 4}} & \frac{\partial L}{\partial x_{1, 5}} \\
\frac{\partial L}{\partial x_{2, 1}} & \frac{\partial L}{\partial x_{2, 2}} & \frac{\partial L}{\partial x_{2, 3}}& \frac{\partial L}{\partial x_{2, 4}} & \frac{\partial L}{\partial x_{2, 5}} \\
\frac{\partial L}{\partial x_{3, 1}} & \frac{\partial L}{\partial x_{3, 2}} & \frac{\partial L}{\partial x_{3, 3}}& \frac{\partial L}{\partial x_{3, 4}} & \frac{\partial L}{\partial x_{3, 5}} \\
\frac{\partial L}{\partial x_{4, 1}} & \frac{\partial L}{\partial x_{4, 2}} & \frac{\partial L}{\partial x_{4, 3}}& \frac{\partial L}{\partial x_{4, 4}} & \frac{\partial L}{\partial x_{4, 5}} \\
\frac{\partial L}{\partial x_{5, 1}} & \frac{\partial L}{\partial x_{5, 2}} & \frac{\partial L}{\partial x_{5, 3}}& \frac{\partial L}{\partial x_{5, 4}} & \frac{\partial L}{\partial x_{5, 5}} \\
\end{bmatrix} = 
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{1, 1} & 0 & \delta_{1, 2} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{2, 1} & 0 & \delta_{2, 2} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
\; conv \;\begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$

&emsp;&emsp;可以发现，在前面提出的三个问题中，有两个问题的答案是和步长stride为1的二维卷积相同的，唯一不同的是，我们需要在误差矩阵的相邻元素之间插入若干0来完成卷积误差的产生。  

###### 误差矩阵插入0的方式  

&emsp;&emsp;很明显，我们唯一需要解决的问题就是如何在误差矩阵中插入0。在这里直接给出结论，那就是**每个相邻的元素之间应该插入（步长stride - 1）个0，或者说每个元素之间的距离是卷积的步长。**因为在这个模型中，唯一和前面的卷积方式不同的变量就是步长stride，那么需要满足的条件也必然和步长有关。  

&emsp;&emsp;当我们在元素之间插入合适数目的0之后，接下来就是在误差矩阵周围填补上合适数目的0层，然后将卷积核旋转180°，最后按照步长为1的方式进行卷积，最后得到应该向前传递的误差矩阵。  这两步和步长stride为1的反向传播算法相同。  

##### 三、参数更新  

&emsp;&emsp;当我们解决了误差的向前传递之后，下一步就是解决参数的更新的问题。和前面的定义一样，假设我们在这一阶段接收到的后方传递过来的误差为$\delta$， 即：  

$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3} \\
\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3} \\
\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3} \\
\end{bmatrix}
$$

&emsp;&emsp;那么根据偏导数求解的链式法则，我们可以有下面的式子：这里以求解$\frac{\partial L}{\partial k_{1, 1}}$ 为例：  

$$
\begin{aligned}
\frac{\partial L}{\partial k_{1, 1}} =& 
\frac{\partial L}{\partial u_{1, 1}} \frac{\partial u_{1, 1}}{k_{1, 1}} + \frac{\partial L}{\partial u_{1, 2}} \frac{\partial u_{1, 2}}{k_{1, 1}} +
\frac{\partial L}{\partial u_{2, 1}} \frac{\partial u_{2, 1}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{2, 2}} \frac{\partial u_{2, 2}}{k_{1, 1}} \\
=&
\delta_{1, 1} \frac{\partial u_{1, 1}}{k_{1, 1}} +
\delta_{1, 2} \frac{\partial u_{1, 2}}{k_{1, 1}} +  
\delta_{2, 1} \frac{\partial u_{2, 1}}{k_{1, 1}} +  
\delta_{2, 2} \frac{\partial u_{2, 2}}{k_{1, 1}} \\
=&
\delta_{1, 1} x_{1, 1} +
\delta_{1, 2} x_{1, 3} + 
\delta_{2, 1} x_{3, 1} +  
\delta_{2, 2} x_{3, 3}
\end{aligned}
$$

&emsp;&emsp;类似地，我们将所有地偏导数信息都求出来，汇总如下：  

$$
\frac{\partial L}{\partial k_{1, 1}} = 
\delta_{1, 1} x_{1, 1} +
\delta_{1, 2} x_{1, 3} + 
\delta_{2, 1} x_{3, 1} +  
\delta_{2, 2} x_{3, 3}
$$
$$
\frac{\partial L}{\partial k_{1, 2}} = 
\delta_{1, 1} x_{1, 2} +
\delta_{1, 2} x_{1, 4} + 
\delta_{2, 1} x_{3, 2} +  
\delta_{2, 2} x_{3, 4}
$$
$$
\frac{\partial L}{\partial k_{1, 3}} = 
\delta_{1, 1} x_{1, 3} +
\delta_{1, 2} x_{1, 5} + 
\delta_{2, 1} x_{3, 3} +  
\delta_{2, 2} x_{3, 5}
$$
$$
\frac{\partial L}{\partial k_{2, 1}} = 
\delta_{1, 1} x_{2, 1} +
\delta_{1, 2} x_{2, 3} + 
\delta_{2, 1} x_{4, 1} +  
\delta_{2, 2} x_{4, 3}
$$
$$
\frac{\partial L}{\partial k_{2, 2}} = 
\delta_{1, 1} x_{2, 2} +
\delta_{1, 2} x_{2, 4} + 
\delta_{2, 1} x_{4, 2} +  
\delta_{2, 2} x_{4, 4}
$$
$$
\frac{\partial L}{\partial k_{2, 3}} = 
\delta_{1, 1} x_{2, 3} +
\delta_{1, 2} x_{2, 5} + 
\delta_{2, 1} x_{4, 3} +  
\delta_{2, 2} x_{4, 5}
$$
$$
\frac{\partial L}{\partial k_{3, 1}} = 
\delta_{1, 1} x_{3, 1} +
\delta_{1, 2} x_{3, 3} + 
\delta_{2, 1} x_{5, 1} +  
\delta_{2, 2} x_{5, 3}
$$
$$
\frac{\partial L}{\partial k_{3, 2}} = 
\delta_{1, 1} x_{3, 2} +
\delta_{1, 2} x_{3, 4} + 
\delta_{2, 1} x_{5, 2} +  
\delta_{2, 2} x_{5, 4}
$$
$$
\frac{\partial L}{\partial k_{3, 3}} = 
\delta_{1, 1} x_{3, 3} +
\delta_{1, 2} x_{3, 5} + 
\delta_{2, 1} x_{5, 3} +  
\delta_{2, 2} x_{5, 5}
$$

$$
\frac{\partial L}{\partial b} = \delta_{1, 1} + \delta_{1, 2} + \delta_{2, 1} + \delta_{2, 2}
$$

&emsp;&emsp;和前面地误差传递类似，我们发现可以在误差矩阵中插入若干个0来和输入矩阵$x$来保持维度上的匹配。即有：  

$$
\frac{\partial L}{\partial k} = [\frac{\partial L}{\partial k_{i, j}}]
= \begin{bmatrix}
x_{1, 1} & x_{1, 2} & x_{1, 3} &x_{1, 4} &x_{1, 5} \\
x_{2, 1} & x_{2, 2} & x_{2, 3} &x_{2, 4} &x_{2, 5} \\
x_{3, 1} & x_{3, 2} & x_{3, 3} &x_{3, 4} &x_{3, 5} \\
x_{4, 1} & x_{4, 2} & x_{4, 3} &x_{4, 4} &x_{4, 5} \\
x_{5, 1} & x_{5, 2} & x_{5, 3} &x_{5, 4} &x_{5, 5} \\
\end{bmatrix} \; conv \; \begin{bmatrix}
\delta_{1, 1} & 0 & \delta_{1, 2} \\
0 & 0 & 0 \\
\delta_{2, 1} & 0 & \delta_{2, 2} \\
\end{bmatrix}
$$

&emsp;&emsp;据此，我们可以发现，**在卷积核参数更新的过程中，我们也需要对误差矩阵进行插入0的操作。而且插入0的方式和误差传递过程中的方式完全相同。**所以，我们可以总结出步长为s的时候卷积反向传播的卷积核参数更新的方法，即：1.首先在接收到的误差矩阵中插入合适数目的0，2.在输入矩阵$x$上应用误差矩阵进行步长为1的卷积，从而得到卷积核的更新梯度。  

&emsp;&emsp;同样，我们由上面的推导可以发现，无论是何种方式的卷积操作，偏置项$b$的更新梯度都是接收到的误差矩阵中的元素之和。  

##### 四、总结  

&emsp;&emsp;我们将上面的求解过程总结如下有：  

| 参数          | 设置         |
| ------------- | ------------ |
| 输入矩阵$x$   | 一个二维矩阵 |
| 输入卷积核$k$ | 一个二维矩阵 |
| 步长$stride$  | 一个正整数s  |
| padding       | VALID        |
| 偏置项$b$     | 一个浮点数   |

&emsp;&emsp;正向传播：  

```text
conv(x, kernel, bias, "VALID")
```

&emsp;&emsp;反向传播：  

```text
conv_backward(error, x, kernel, bias):
	# 计算传递给下一层的误差
	1.在接收到的error矩阵的矩阵中插入合适数目的0，使得每个元素之间的0的数目为(stride - 1)
	2.在error周围填补上合适数目的0
	3.将kernel旋转180°
	4.将填补上0的误差和旋转之后的kernel进行步长为1的卷积，从而得到传递给下一层的误差new_error。
	
	# 更新参数
	1.在接收到的error矩阵的矩阵中插入合适数目的0，使得每个元素之间的0的数目为(stride - 1)
	2.将输入矩阵x和插入0之后的误差矩阵error进行步长为1的卷积，得到kernel的更新梯度
	3.将上一层传递来的误差矩阵error所有元素求和，得到bias的更新梯度
	4.kernel := kernel - 学习率 * kernel的更新梯度
	5.bias := bias - 学习率 * bias的更新梯度
	
	# 返回误差，用以传递到下一层
	return new_error
```





