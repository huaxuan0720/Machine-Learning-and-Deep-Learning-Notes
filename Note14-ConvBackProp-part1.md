---
title: 步长stride为1的二维卷积方法的反向传播算法
date: 2019-05-24 12:32:01
tags: [卷积, 反向传播, 深度学习]
category: 深度学习
toc: true
thumbnail: gallery/DeepLearning.jpg
---

##### 前言  

&emsp;&emsp;近年来，深度学习的快速发展带来了一系列喜人的成果，不管是在图像领域还是在NLP领域，深度学习都显示了其极其强大的能力。而深度学习之所以可以进行大规模的参数运算，反向传播算法功不可没，可以说，没有反向传播算法，深度学习就不可能得以快速发展，因此在此之前，有必要了解一下反向传播算法的具体原理和公式推导。请注意：这里的所有推导过程都只是针对当前设置的参数信息，并不具有一般性，但是所有的推导过程可以推导到一般的运算，因此以下给出的并不是反向传播算法的严格证明，不涉及十分复杂的公式推导，争取可以以一种简单的方式来理解卷积的反向传播。希望可以很好的帮助理解反向传播算法。  

&emsp;&emsp;在之前曾经见到的说明过全连接层的反向传播，因此，这一次主要是专注于卷积层的反向传播。  

&emsp;&emsp;需要注意的是，在本文中，所有的正向传播过程中，卷积的步长stride均固定为1。  

<!--more-->

##### 一、参数设置  
&emsp;&emsp;在前面的全连接层的反向传播算法的推导中，其实可以发现，反向传播算法的主要核心功能就是两个，一个是进行误差的向前传播，一个是进行参数的更新，当解决了这两个问题之后，某一个特定操作的反向传播算法就得到了解决。因此，我们先从一个简单具体的实例入手。  

&emsp;&emsp;由于卷积往往可以看作一个二维平面上的操作，那么我们就先设定我们对一个二维数据矩阵进行卷积，卷积核则也是一个二维矩阵，步长参数我们首先设置为1，在以后的说明中，步长可以设定为其他的数值。按照TensorFlow定义的卷积格式，这里的padding我们默认均为VALID，事实上，如果设置为SAME，也可以通过填补0的方式改变成VALID。  

&emsp;&emsp;这里我们设置我们的数据矩阵（记作$x$）大小为5x5，卷积核（记作$k$）大小为3x3，由于步长是1，因此，卷积之后获得的结果是一个3x3大小的数据矩阵（不妨我们记作$u$）。偏置项我们记为$b$，将和卷积之后的矩阵进行相加。  
&emsp;&emsp;我们的参数汇总如下：  


|参数|设置|
|---|---|
|输入矩阵$x$|一个二维矩阵，大小为5x5|
|输入卷积核$k$|一个二维矩阵，大小为3x3|
|步长$stride$|始终为1|
|padding|VALID|
|偏置项$b$|一个浮点数|

&emsp;&emsp;我们定义卷积操作的符号为$conv$，我们可以将卷积表示为：  

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
u_{1, 1} & u_{1, 2} & u_{1, 3}\\
u_{2, 1} & u_{2, 2} & u_{2, 3}\\
u_{3, 1} & u_{3, 2} & u_{3, 3}\\
\end{bmatrix}
$$

&emsp;&emsp;我们将结果$u$继续展开，可以得到下面的庞大的矩阵：  

$$
\begin{bmatrix}
u_{1, 1} & u_{1, 2} & u_{1, 3}\\
u_{2, 1} & u_{2, 2} & u_{2, 3}\\
u_{3, 1} & u_{3, 2} & u_{3, 3}\\
\end{bmatrix}
= \\
\begin{bmatrix}
\begin{matrix}
x_{1, 1}k_{1, 1} + x_{1, 2}k_{1, 2} +x_{1, 3}k_{1, 3} + \\
x_{2, 1}k_{2, 1} + x_{2, 2}k_{2, 2} +x_{2, 3}k_{2, 3} + \\
x_{3, 1}k_{3, 1} + x_{3, 2}k_{3, 2} +x_{3, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{1, 2}k_{1, 1} + x_{1, 3}k_{1, 2} +x_{1, 4}k_{1, 3} + \\
x_{2, 2}k_{2, 1} + x_{2, 3}k_{2, 2} +x_{2, 4}k_{2, 3} + \\
x_{3, 2}k_{3, 1} + x_{3, 3}k_{3, 2} +x_{3, 4}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{1, 3}k_{1, 1} + x_{1, 4}k_{1, 2} +x_{1, 5}k_{1, 3} + \\
x_{2, 3}k_{2, 1} + x_{2, 4}k_{2, 2} +x_{2, 5}k_{2, 3} + \\
x_{3, 3}k_{3, 1} + x_{3, 4}k_{3, 2} +x_{3, 5}k_{3, 3} + b \\
\end{matrix} \\ \\
\begin{matrix}
x_{2, 1}k_{1, 1} + x_{2, 2}k_{1, 2} +x_{2, 3}k_{1, 3} + \\
x_{3, 1}k_{2, 1} + x_{3, 2}k_{2, 2} +x_{3, 3}k_{2, 3} + \\
x_{4, 1}k_{3, 1} + x_{4, 2}k_{3, 2} +x_{4, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{2, 2}k_{1, 1} + x_{2, 3}k_{1, 2} +x_{2, 4}k_{1, 3} + \\
x_{3, 2}k_{2, 1} + x_{3, 3}k_{2, 2} +x_{3, 4}k_{2, 3} + \\
x_{4, 2}k_{3, 1} + x_{4, 3}k_{3, 2} +x_{4, 4}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{2, 3}k_{1, 1} + x_{2, 4}k_{1, 2} +x_{2, 5}k_{1, 3} + \\
x_{3, 3}k_{2, 1} + x_{3, 4}k_{2, 2} +x_{3, 5}k_{2, 3} + \\
x_{4, 3}k_{3, 1} + x_{4, 4}k_{3, 2} +x_{4, 5}k_{3, 3} + b \\
\end{matrix} \\ \\
\begin{matrix}
x_{3, 1}k_{1, 1} + x_{3, 2}k_{1, 2} +x_{3, 3}k_{1, 3} + \\
x_{4, 1}k_{2, 1} + x_{4, 2}k_{2, 2} +x_{4, 3}k_{2, 3} + \\
x_{5, 1}k_{3, 1} + x_{5, 2}k_{3, 2} +x_{5, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{3, 2}k_{1, 1} + x_{3, 3}k_{1, 2} +x_{3, 4}k_{1, 3} + \\
x_{4, 2}k_{2, 1} + x_{4, 3}k_{2, 2} +x_{4, 4}k_{2, 3} + \\
x_{5, 2}k_{3, 1} + x_{5, 3}k_{3, 2} +x_{5, 4}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{3, 3}k_{1, 1} + x_{3, 4}k_{1, 2} +x_{3, 5}k_{1, 3} + \\
x_{4, 3}k_{2, 1} + x_{4, 4}k_{2, 2} +x_{4, 5}k_{2, 3} + \\
x_{5, 3}k_{3, 1} + x_{5, 4}k_{3, 2} +x_{5, 5}k_{3, 3} + b \\
\end{matrix} \\
\end{bmatrix}
$$

##### 二、误差前向传递  

&emsp;&emsp;在前面已经完整的表示出了卷积的所有操作，下面我们来进行误差传递。  

&emsp;&emsp;我们对上面的所有的输入进行求解偏导数的操作，我们可以得到下面的一张表格，每一列表示的是一个特定的输出 $\partial u_{i, j}$，每一行表示的是一个特定的输入值$\partial x_{p, k}$，行与列相交的地方表示的就是二者相除的结果，表示的是输出对于输入的偏导数，即$\frac{\partial u_{i, j}}{\partial x_{p, k}}$。于是，表格如下：  


|                     | $\partial u_{1, 1}$ | $\partial u_{1, 2}$ | $\partial u_{1, 3}$ | $\partial u_{2, 1}$ | $\partial u_{2, 2}$ | $\partial u_{2, 3}$ | $\partial u_{3, 1}$ | $\partial u_{3, 2}$ | $\partial u_{3, 3}$ |
| ------------------- | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :-----------------: |
| $\partial x_{1, 1}$ |     $k_{1, 1}$      |          0          |          0          |          0          |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{1, 2}$ |     $k_{1, 2}$      |     $k_{1, 1}$      |          0          |          0          |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{1, 3}$ |     $k_{1, 3}$      |     $k_{1, 2}$      |     $k_{1, 1}$      |          0          |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{1, 4}$ |          0          |     $k_{1, 3}$      |     $k_{1, 2}$      |          0          |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{1, 5}$ |          0          |          0          |     $k_{1, 3}$      |          0          |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{2, 1}$ |     $k_{2, 1}$      |          0          |          0          |     $k_{1, 1}$      |          0          |          0          |          0          |          0          |          0          |
| $\partial x_{2, 2}$ |     $k_{2, 2}$      |     $k_{2, 1}$      |          0          |     $k_{1, 2}$      |     $k_{1, 1}$      |          0          |          0          |          0          |          0          |
| $\partial x_{2, 3}$ |     $k_{2, 3}$      |     $k_{2, 2}$      |     $k_{2, 1}$      |     $k_{1, 3}$      |     $k_{1, 2}$      |     $k_{1, 1}$      |          0          |          0          |          0          |
| $\partial x_{2, 4}$ |          0          |     $k_{2, 3}$      |     $k_{2, 2}$      |          0          |     $k_{1, 3}$      |     $k_{1, 2}$      |          0          |          0          |          0          |
| $\partial x_{2, 5}$ |          0          |          0          |     $k_{2, 3}$      |          0          |          0          |     $k_{1, 3}$      |          0          |          0          |          0          |
| $\partial x_{3, 1}$ |     $k_{3, 1}$      |          0          |          0          |     $k_{2, 1}$      |          0          |          0          |     $k_{1, 1}$      |          0          |          0          |
| $\partial x_{3, 2}$ |     $k_{3, 2}$      |     $k_{3, 1}$      |          0          |     $k_{2, 2}$      |     $k_{2, 1}$      |          0          |     $k_{1, 2}$      |     $k_{1, 1}$      |          0          |
| $\partial x_{3, 3}$ |     $k_{3, 3}$      |     $k_{3, 2}$      |     $k_{3, 1}$      |     $k_{2, 3}$      |     $k_{2, 2}$      |     $k_{2, 1}$      |     $k_{1, 3}$      |     $k_{1, 2}$      |     $k_{1, 1}$      |
| $\partial x_{3, 4}$ |          0          |     $k_{3, 3}$      |     $k_{3, 2}$      |          0          |     $k_{2, 3}$      |     $k_{2, 2}$      |          0          |     $k_{1, 3}$      |     $k_{1, 2}$      |
| $\partial x_{3, 5}$ |          0          |          0          |     $k_{3, 3}$      |          0          |          0          |     $k_{2, 3}$      |          0          |          0          |     $k_{1, 3}$      |
| $\partial x_{4, 1}$ |          0          |          0          |          0          |     $k_{3, 1}$      |          0          |          0          |     $k_{2, 1}$      |          0          |          0          |
| $\partial x_{4, 2}$ |          0          |          0          |          0          |     $k_{3, 2}$      |     $k_{3, 1}$      |          0          |     $k_{2, 2}$      |     $k_{2, 1}$      |          0          |
| $\partial x_{4, 3}$ |          0          |          0          |          0          |     $k_{3, 3}$      |     $k_{3, 2}$      |     $k_{3, 1}$      |     $k_{2, 3}$      |     $k_{2, 2}$      |     $k_{2, 1}$      |
| $\partial x_{4, 4}$ |          0          |          0          |          0          |          0          |     $k_{3, 3}$      |     $k_{3, 2}$      |          0          |     $k_{2, 3}$      |     $k_{2, 2}$      |
| $\partial x_{4, 5}$ |          0          |          0          |          0          |          0          |          0          |     $k_{3, 3}$      |          0          |          0          |     $k_{2, 3}$      |
| $\partial x_{5, 1}$ |          0          |          0          |          0          |          0          |          0          |          0          |     $k_{3, 1}$      |          0          |          0          |
| $\partial x_{5, 2}$ |          0          |          0          |          0          |          0          |          0          |          0          |     $k_{3, 2}$      |     $k_{3, 1}$      |          0          |
| $\partial x_{5, 3}$ |          0          |          0          |          0          |          0          |          0          |          0          |     $k_{3, 3}$      |     $k_{3, 2}$      |     $k_{3, 1}$      |
| $\partial x_{5, 4}$ |          0          |          0          |          0          |          0          |          0          |          0          |          0          |     $k_{3, 3}$      |     $k_{3, 2}$      |
| $\partial x_{5, 5}$ |          0          |          0          |          0          |          0          |          0          |          0          |          0          |          0          |     $k_{3, 3}$      |

&emsp;&emsp;可以看出，数据都是很规律的进行着重复。  

&emsp;&emsp;我们假设后面传递过来的误差是 $\delta$ ，即：  

$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3} \\
\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3} \\
\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3} \\
\end{bmatrix}
$$

&emsp;&emsp;其中，$\delta_{i, j} = \frac{\partial L}{\partial u_{i, j}}$，误差分别对应于每一个输出项。这里的$L$表示的是最后的Loss损失。我们的目的就是希望这个损失尽可能小。那么，根据求导的链式法则，我们有：  

$$
\frac{\partial L}{\partial x_{p, k}} = \sum^{3}_{i = 1} \sum^{3}_{j = 1} \frac{\partial L}{\partial u_{i, j}} \cdot \frac{\partial u_{i, j}}{\partial x_{p, k}} = \sum^{3}_{i = 1} \sum^{3}_{j = 1} \delta_{i, j} \cdot \frac{\partial u_{i, j}}{\partial x_{p, k}}
$$

&emsp;&emsp;根据这个公式，我们可以有：  

$$
\frac{\partial L}{\partial x_{1, 1}} = \delta_{1, 1} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{1, 2}} = \delta_{1, 1} \cdot k_{1, 2} + \delta_{1, 2} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{1, 3}} = \delta_{1, 1} \cdot k_{1, 3} + \delta_{1, 2} \cdot k_{1, 2} + \delta_{1, 3} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{1, 4}} = \delta_{1, 2} \cdot k_{1, 3} + \delta_{1, 3} \cdot k_{1, 2}
$$
$$
\frac{\partial L}{\partial x_{1, 5}} = \delta_{1, 3} \cdot k_{1, 3}
$$
$$
\frac{\partial L}{\partial x_{2, 1}} = \delta_{1, 1} \cdot k_{2, 1} +  \delta_{2, 1} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{2, 2}} = \delta_{1, 1} \cdot k_{2, 2} + \delta_{1, 2} \cdot k_{2, 1} + \delta_{2, 1} \cdot k_{1,2}+ \delta_{2, 2} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{2, 3}} = \delta_{1, 1} \cdot k_{2, 3} + \delta_{1, 2} \cdot k_{2, 2} + \delta_{1, 3} \cdot k_{2, 1} + \delta_{2, 1} \cdot k_{1,3}+ \delta_{2,2} \cdot k_{1, 2} +\delta_{2,3} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{2, 4}} = \delta_{1, 2} \cdot k_{2, 3} + \delta_{1, 3} \cdot k_{2, 2} + \delta_{2, 2} \cdot k_{1,3}+ \delta_{2, 3} \cdot k_{1, 2}
$$
$$
\frac{\partial L}{\partial x_{2, 5}} = \delta_{1, 3} \cdot k_{2, 3} +  \delta_{2, 3} \cdot k_{1, 3}
$$
$$
\frac{\partial L}{\partial x_{3, 1}} = \delta_{1, 1} \cdot k_{3, 1} + \delta_{2, 1} \cdot k_{2, 1} + \delta_{3, 1} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{3, 2}} = \delta_{1, 1} \cdot k_{3, 2} + \delta_{1, 2} \cdot k_{3, 1} + \delta_{2, 1} \cdot k_{2, 2} + \delta_{2, 2} \cdot k_{2, 1} + \delta_{3, 1} \cdot k_{1, 2} + \delta_{3, 2} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{3, 3}} = \delta_{1, 1} \cdot k_{3, 3} + \delta_{1, 2} \cdot k_{3, 2} + \delta_{1, 3} \cdot k_{3, 1} + \delta_{2, 1} \cdot k_{2, 3} + \delta_{2, 2} \cdot k_{2, 2} + \delta_{2, 3} \cdot k_{2, 1} + \delta_{3, 1} \cdot k_{1, 3} + \delta_{3, 2} \cdot k_{1, 2} + \delta_{3, 3} \cdot k_{1, 1}
$$
$$
\frac{\partial L}{\partial x_{3, 4}} = \delta_{1, 2} \cdot k_{3, 3} + \delta_{1, 3} \cdot k_{3, 2} + \delta_{2, 2} \cdot k_{2, 3} + \delta_{2, 3} \cdot k_{2, 2} + \delta_{3, 2} \cdot k_{1, 3} + \delta_{3, 3} \cdot k_{1, 2}
$$
$$
\frac{\partial L}{\partial x_{3, 5}} = \delta_{1, 3} \cdot k_{3, 3} + \delta_{2, 3} \cdot k_{2, 3} + \delta_{3, 3} \cdot k_{1, 3}
$$
$$
\frac{\partial L}{\partial x_{4, 1}} = \delta_{2, 1} \cdot k_{3, 1} + \delta_{3, 1} \cdot k_{2, 1}
$$
$$
\frac{\partial L}{\partial x_{4, 2}} = \delta_{2, 1} \cdot k_{3, 2} + \delta_{2, 2} \cdot k_{3, 1} + \delta_{3, 1} \cdot k_{2, 2} + \delta_{3, 2} \cdot k_{2, 1}
$$
$$
\frac{\partial L}{\partial x_{4, 3}} = \delta_{2, 1} \cdot k_{3, 3} + \delta_{2, 2} \cdot k_{3, 2} + \delta_{2, 3} \cdot k_{3, 1} + \delta_{3, 1} \cdot k_{2, 3} + \delta_{3, 2} \cdot k_{2, 2} + \delta_{3, 3} \cdot k_{2, 1}
$$
$$
\frac{\partial L}{\partial x_{4, 4}} = \delta_{2, 2} \cdot k_{3, 3} + \delta_{2, 3} \cdot k_{3, 2} + \delta_{3, 2} \cdot k_{2, 3} + \delta_{3, 3} \cdot k_{2, 2}
$$
$$
\frac{\partial L}{\partial x_{4, 5}} = \delta_{2, 3} \cdot k_{3, 3} + \delta_{3, 3} \cdot k_{2, 3}
$$
$$
\frac{\partial L}{\partial x_{5, 1}} = \delta_{3, 1} \cdot k_{3, 1}
$$
$$
\frac{\partial L}{\partial x_{5, 2}} = \delta_{3, 1} \cdot k_{3, 2} + \delta_{3, 2} \cdot k_{3, 1}
$$
$$
\frac{\partial L}{\partial x_{5, 3}} = \delta_{3, 1} \cdot k_{3, 3} +\delta_{3, 2} \cdot k_{3, 2} + \delta_{3, 3} \cdot k_{3, 1}
$$
$$
\frac{\partial L}{\partial x_{5, 4}} = \delta_{3, 2} \cdot k_{3, 3} + \delta_{3, 3} \cdot k_{3, 2}
$$
$$
\frac{\partial L}{\partial x_{5, 5}} = \delta_{3, 3} \cdot k_{3, 3}
$$

&emsp;&emsp;以上的式子虽然多，烦，一不小心就容易出错，但是每一个式子都是很简单的相乘相加，因此，我们考虑使用向量化或者矩阵化的表达方式。  

&emsp;&emsp;为了更好的进行矩阵化表达，我们将$\frac{\partial L}{\partial x_{3, 3}}$单独拿出来看，我们发现，这个式子可以变成两个相同的矩阵进行卷积（由于使用的是padding为VALID的模式，因此，在这种情况下，步长stride信息可有可无。），即：  

$$
\frac{\partial L}{\partial x_{3, 3}} = \begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3} \\
\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3} \\
\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3} \\
\end{bmatrix} \; conv \; \begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$

&emsp;&emsp;进一步，我们发现，以上所有的式子的构成元素都包含在上面的两个矩阵中。  我们记右侧的全部由卷积核元素构成的矩阵为$k'$  

&emsp;&emsp;下面的一个步骤需要一点观察技巧了，如果在$\delta$矩阵的上下左右同时填上两层0，变成如下的形式：   

$$
\begin{bmatrix}
0 &0 &0&0&0&0&0\\
0 &0 &0&0&0&0&0\\
0 &0 &\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3}&0 &0 \\
0 &0 &\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3}&0 &0 \\
0 &0 &\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3}&0 &0 \\
0 &0 &0&0&0&0&0\\
0 &0 &0&0&0&0&0\\
\end{bmatrix}
$$

&emsp;&emsp;在此基础上，我们利用矩阵$k'$对上式进行卷积，该卷积的步长stride为1，可以得到一个和原始的输入矩阵相同大小的矩阵，不妨记该矩阵作$x’$。  

&emsp;&emsp;接着，我们对之前求得的25个式子按照对应的顺序进行排列，记作$x''$，于是有：  

$$
x'' = \begin{bmatrix}
\frac{\partial L}{\partial x_{1, 1}} & \frac{\partial L}{\partial x_{1, 2}} & \frac{\partial L}{\partial x_{1, 3}}& \frac{\partial L}{\partial x_{1, 4}} & \frac{\partial L}{\partial x_{1, 5}} \\
\frac{\partial L}{\partial x_{2, 1}} & \frac{\partial L}{\partial x_{2, 2}} & \frac{\partial L}{\partial x_{2, 3}}& \frac{\partial L}{\partial x_{2, 4}} & \frac{\partial L}{\partial x_{2, 5}} \\
\frac{\partial L}{\partial x_{3, 1}} & \frac{\partial L}{\partial x_{3, 2}} & \frac{\partial L}{\partial x_{3, 3}}& \frac{\partial L}{\partial x_{3, 4}} & \frac{\partial L}{\partial x_{3, 5}} \\
\frac{\partial L}{\partial x_{4, 1}} & \frac{\partial L}{\partial x_{4, 2}} & \frac{\partial L}{\partial x_{4, 3}}& \frac{\partial L}{\partial x_{4, 4}} & \frac{\partial L}{\partial x_{4, 5}} \\
\frac{\partial L}{\partial x_{5, 1}} & \frac{\partial L}{\partial x_{5, 2}} & \frac{\partial L}{\partial x_{5, 3}}& \frac{\partial L}{\partial x_{5, 4}} & \frac{\partial L}{\partial x_{5, 5}} \\
\end{bmatrix}
$$

&emsp;&emsp;经过计算，我们可以发现，$x'$和$x''$正好相等。即：  

$$
\begin{bmatrix}
\frac{\partial L}{\partial x_{1, 1}} & \frac{\partial L}{\partial x_{1, 2}} & \frac{\partial L}{\partial x_{1, 3}}& \frac{\partial L}{\partial x_{1, 4}} & \frac{\partial L}{\partial x_{1, 5}} \\
\frac{\partial L}{\partial x_{2, 1}} & \frac{\partial L}{\partial x_{2, 2}} & \frac{\partial L}{\partial x_{2, 3}}& \frac{\partial L}{\partial x_{2, 4}} & \frac{\partial L}{\partial x_{2, 5}} \\
\frac{\partial L}{\partial x_{3, 1}} & \frac{\partial L}{\partial x_{3, 2}} & \frac{\partial L}{\partial x_{3, 3}}& \frac{\partial L}{\partial x_{3, 4}} & \frac{\partial L}{\partial x_{3, 5}} \\
\frac{\partial L}{\partial x_{4, 1}} & \frac{\partial L}{\partial x_{4, 2}} & \frac{\partial L}{\partial x_{4, 3}}& \frac{\partial L}{\partial x_{4, 4}} & \frac{\partial L}{\partial x_{4, 5}} \\
\frac{\partial L}{\partial x_{5, 1}} & \frac{\partial L}{\partial x_{5, 2}} & \frac{\partial L}{\partial x_{5, 3}}& \frac{\partial L}{\partial x_{5, 4}} & \frac{\partial L}{\partial x_{5, 5}} \\
\end{bmatrix} = \begin{bmatrix}
0 &0 &0&0&0&0&0\\
0 &0 &0&0&0&0&0\\
0 &0 &\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3}&0 &0 \\
0 &0 &\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3}&0 &0 \\
0 &0 &\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3}&0 &0 \\
0 &0 &0&0&0&0&0\\
0 &0 &0&0&0&0&0\\
\end{bmatrix} \; conv \; \begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$

&emsp;&emsp;在这个卷积操作中，步长stride为1。  

&emsp;&emsp;所以我们发现，在卷积操作中误差的传递主要是利用该卷积的卷积核（经过一定的变换）对传递而来的误差进行卷积来完成的。所以，我们要解决的问题又两个，一个是卷积核的变换是什么样子的，另一个就是需要在传递来的误差上下左右填补多少0。  


###### 卷积核的变换  

&emsp;&emsp;在前面，我们发现误差传递的时候使用的卷积核和正向传播时使用的略有不同，事实上，**在误差传递的时候，我们使用的卷积核是正向传播时使用的卷积核的中心对称矩阵，抑或是将正向传播使用的矩阵旋转180°之后就得到了误差传递时使用的矩阵**。在这里，并不需要严格证明这一结果，只需要知道需要这么做即可。  


###### 填补0

&emsp;&emsp;另一个问题是我们需要在传递过来的误差矩阵周围填补多少0。我们在这里假设输入矩阵是一个正方形矩阵，卷积核也是一个正方形矩阵，输入矩阵的长宽均为$n$，卷积核的长宽均为$k$，步长为1，则输出的矩阵长宽为$m = n - ( k - 1)$。假设经过填补0之后的误差矩阵长宽均为$x$，因为我们需要对卷积核进行旋转180°，所以卷积核长宽保持不变，所以有：  

$$
x - (k - 1) = n
$$

&emsp;&emsp;又有：  

$$
\because m = n - (k - 1) \\
\therefore n = m + (k - 1) \\
\therefore x - (k - 1) = m + (k - 1) \\
\therefore x = m + 2 * (k - 1)
$$

&emsp;&emsp;因此，上下左右需要填补$k - 1$层0。  

&emsp;&emsp;在这里只讨论了正方形的输入矩阵和正方形的卷积核，但是这一结论很容易推广到任意尺寸的输入矩阵和卷积核，这里就不再赘述。  

&emsp;&emsp;至此，我们就解决了在步长stride为1的卷积过程中的误差传递的问题。下面就是解决参数更新的问题了。  

##### 三、参数更新  

&emsp;&emsp;和误差传递类似，我们需要对每一个可以更新的参数求解偏导数，和前面的定义一样，假设我们在这一阶段接收到的后方传递过来的误差为$\delta$， 即：  

$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3} \\
\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3} \\
\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3} \\
\end{bmatrix}
$$

那么根据偏导数求解的链式法则，我们可以有下面的式子：这里以求解$\frac{\partial L}{\partial k_{1, 1}}$ 为例：  

$$
\begin{aligned}
\frac{\partial L}{\partial k_{1, 1}} =& 
\frac{\partial L}{\partial u_{1, 1}} \frac{\partial u_{1, 1}}{k_{1, 1}} + \frac{\partial L}{\partial u_{1, 2}} \frac{\partial u_{1, 2}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{1, 3}} \frac{\partial u_{1, 3}}{k_{1, 1}} +  \\
&\frac{\partial L}{\partial u_{2, 1}} \frac{\partial u_{2, 1}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{2, 2}} \frac{\partial u_{2, 2}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{2, 3}} \frac{\partial u_{2, 3}}{k_{1, 1}} +  \\
&\frac{\partial L}{\partial u_{3, 1}} \frac{\partial u_{3, 1}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{3, 2}} \frac{\partial u_{3, 2}}{k_{1, 1}} +  
\frac{\partial L}{\partial u_{3, 3}} \frac{\partial u_{3, 3}}{k_{1, 1}} \\
=&
\delta_{1, 1} \frac{\partial u_{1, 1}}{k_{1, 1}} +
\delta_{1, 2} \frac{\partial u_{1, 2}}{k_{1, 1}} +  
\delta_{1, 3} \frac{\partial u_{1, 3}}{k_{1, 1}} +  \\
&\delta_{2, 1} \frac{\partial u_{2, 1}}{k_{1, 1}} +  
\delta_{2, 2} \frac{\partial u_{2, 2}}{k_{1, 1}} +  
\delta_{2, 3} \frac{\partial u_{2, 3}}{k_{1, 1}} +  \\
&\delta_{3, 1} \frac{\partial u_{3, 1}}{k_{1, 1}} +  
\delta_{3, 2} \frac{\partial u_{3, 2}}{k_{1, 1}} +  
\delta_{3, 3} \frac{\partial u_{3, 3}}{k_{1, 1}} \\
=&
\delta_{1, 1} x_{1, 1} +
\delta_{1, 2} x_{1, 2} +  
\delta_{1, 3} x_{1, 3} +  
\delta_{2, 1} x_{2, 1} +  
\delta_{2, 2} x_{2, 2} +  
\delta_{2, 3} x_{2, 3} +  
\delta_{3, 1} x_{3, 1} +  
\delta_{3, 2} x_{3, 2} +  
\delta_{3, 3} x_{3, 3}
\end{aligned}
$$

&emsp;&emsp;类似地，我们可以求出剩下的所有的偏导数，这里我们汇总如下：  

$$
\frac{\partial L}{\partial k_{1, 1}} = 
\delta_{1, 1} x_{1, 1} +
\delta_{1, 2} x_{1, 2} +  
\delta_{1, 3} x_{1, 3} +  
\delta_{2, 1} x_{2, 1} +  
\delta_{2, 2} x_{2, 2} +  
\delta_{2, 3} x_{2, 3} +  
\delta_{3, 1} x_{3, 1} +  
\delta_{3, 2} x_{3, 2} +  
\delta_{3, 3} x_{3, 3}
$$
$$
\frac{\partial L}{\partial k_{1, 2}} = 
\delta_{1, 1} x_{1, 2} +
\delta_{1, 2} x_{1, 3} +  
\delta_{1, 3} x_{1, 4} +  
\delta_{2, 1} x_{2, 2} +  
\delta_{2, 2} x_{2, 3} +  
\delta_{2, 3} x_{2, 4} +  
\delta_{3, 1} x_{3, 2} +  
\delta_{3, 2} x_{3, 3} +  
\delta_{3, 3} x_{3, 4}
$$
$$
\frac{\partial L}{\partial k_{1, 3}} = 
\delta_{1, 1} x_{1, 3} +
\delta_{1, 2} x_{1, 4} +  
\delta_{1, 3} x_{1, 5} +  
\delta_{2, 1} x_{2, 3} +  
\delta_{2, 2} x_{2, 4} +  
\delta_{2, 3} x_{2, 5} +  
\delta_{3, 1} x_{3, 3} +  
\delta_{3, 2} x_{3, 4} +  
\delta_{3, 3} x_{3, 5}
$$
$$
\frac{\partial L}{\partial k_{2, 1}} = 
\delta_{1, 1} x_{2, 1} +
\delta_{1, 2} x_{2, 2} +  
\delta_{1, 3} x_{2, 3} +  
\delta_{2, 1} x_{3, 1} +  
\delta_{2, 2} x_{3, 2} +  
\delta_{2, 3} x_{3, 3} +  
\delta_{3, 1} x_{4, 1} +  
\delta_{3, 2} x_{4, 2} +  
\delta_{3, 3} x_{4, 3}
$$
$$
\frac{\partial L}{\partial k_{2, 2}} = 
\delta_{1, 1} x_{2, 2} +
\delta_{1, 2} x_{2, 3} +  
\delta_{1, 3} x_{2, 4} +  
\delta_{2, 1} x_{3, 2} +  
\delta_{2, 2} x_{3, 3} +  
\delta_{2, 3} x_{3, 4} +  
\delta_{3, 1} x_{4, 2} +  
\delta_{3, 2} x_{4, 3} +  
\delta_{3, 3} x_{4, 4}
$$
$$
\frac{\partial L}{\partial k_{2, 3}} = 
\delta_{1, 1} x_{2, 3} +
\delta_{1, 2} x_{2, 4} +  
\delta_{1, 3} x_{2, 5} +  
\delta_{2, 1} x_{3, 3} +  
\delta_{2, 2} x_{3, 4} +  
\delta_{2, 3} x_{3, 5} +  
\delta_{3, 1} x_{4, 3} +  
\delta_{3, 2} x_{4, 4} +  
\delta_{3, 3} x_{4, 5}
$$
$$
\frac{\partial L}{\partial k_{3, 1}} = 
\delta_{1, 1} x_{3, 1} +
\delta_{1, 2} x_{3, 2} +  
\delta_{1, 3} x_{3, 3} +  
\delta_{2, 1} x_{4, 1} +  
\delta_{2, 2} x_{4, 2} +  
\delta_{2, 3} x_{4, 3} +  
\delta_{3, 1} x_{5, 1} +  
\delta_{3, 2} x_{5, 2} +  
\delta_{3, 3} x_{5, 3}
$$
$$
\frac{\partial L}{\partial k_{3, 2}} = 
\delta_{1, 1} x_{3, 2} +
\delta_{1, 2} x_{3, 3} +  
\delta_{1, 3} x_{3, 4} +  
\delta_{2, 1} x_{4, 2} +  
\delta_{2, 2} x_{4, 3} +  
\delta_{2, 3} x_{4, 4} +  
\delta_{3, 1} x_{5, 2} +  
\delta_{3, 2} x_{5, 3} +  
\delta_{3, 3} x_{5, 4}
$$
$$
\frac{\partial L}{\partial k_{3, 3}} = 
\delta_{1, 1} x_{3, 3} +
\delta_{1, 2} x_{3, 4} +  
\delta_{1, 3} x_{3, 5} +  
\delta_{2, 1} x_{4, 3} +  
\delta_{2, 2} x_{4, 4} +  
\delta_{2, 3} x_{4, 5} +  
\delta_{3, 1} x_{5, 3} +  
\delta_{3, 2} x_{5, 4} +  
\delta_{3, 3} x_{5, 5}
$$

$$
\frac{\partial L}{\partial b} = 
\delta_{1, 1}+
\delta_{1, 2}+  
\delta_{1, 3}+  
\delta_{2, 1}+  
\delta_{2, 2}+  
\delta_{2, 3}+  
\delta_{3, 1}+  
\delta_{3, 2}+  
\delta_{3, 3}
$$

&emsp;&emsp;同样，我们将上面的偏导数信息整理一下，按照每个元素对应的位置进行排列，于是，我们有：  

$$
\frac{\partial L}{\partial k} = [\frac{\partial L}{\partial k_{i, j}}] = 
\begin{bmatrix}
\frac{\partial L}{\partial k_{1, 1}} & \frac{\partial L}{\partial k_{1, 2}} & \frac{\partial L}{\partial k_{1, 3}} \\
\frac{\partial L}{\partial k_{2, 1}} & \frac{\partial L}{\partial k_{2, 2}} & \frac{\partial L}{\partial k_{2, 3}} \\
\frac{\partial L}{\partial k_{3, 1}} & \frac{\partial L}{\partial k_{3, 2}} & \frac{\partial L}{\partial k_{3, 3}} \\
\end{bmatrix}
$$

&emsp;&emsp;当我们这么整理之后，可以发现，这个矩阵可以拆解成两个矩阵的步长为1的卷积，即有：  

$$
\begin{bmatrix}
\frac{\partial L}{\partial k_{1, 1}} & \frac{\partial L}{\partial k_{1, 2}} & \frac{\partial L}{\partial k_{1, 3}} \\
\frac{\partial L}{\partial k_{2, 1}} & \frac{\partial L}{\partial k_{2, 2}} & \frac{\partial L}{\partial k_{2, 3}} \\
\frac{\partial L}{\partial k_{3, 1}} & \frac{\partial L}{\partial k_{3, 2}} & \frac{\partial L}{\partial k_{3, 3}} \\
\end{bmatrix} = 
\begin{bmatrix}
x_{1, 1} & x_{1, 2} & x_{1, 3} &x_{1, 4} &x_{1, 5} \\
x_{2, 1} & x_{2, 2} & x_{2, 3} &x_{2, 4} &x_{2, 5} \\
x_{3, 1} & x_{3, 2} & x_{3, 3} &x_{3, 4} &x_{3, 5} \\
x_{4, 1} & x_{4, 2} & x_{4, 3} &x_{4, 4} &x_{4, 5} \\
x_{5, 1} & x_{5, 2} & x_{5, 3} &x_{5, 4} &x_{5, 5} \\
\end{bmatrix} \; conv \; \begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} & \delta_{1, 3} \\
\delta_{2, 1} & \delta_{2, 2} & \delta_{2, 3} \\
\delta_{3, 1} & \delta_{3, 2} & \delta_{3, 3} \\
\end{bmatrix}
$$

&emsp;&emsp;因此，我们可以总结出，权重的梯度就是输入矩阵和误差矩阵进行步长为1卷积产生的结果矩阵。  

&emsp;&emsp;对于偏置项的梯度$\frac{\partial L}{\partial b}$则是全部的误差矩阵的元素的和。  


##### 四、总结  

&emsp;&emsp;我们将上面的求解过程总结如下有：  

| 参数          | 设置         |
| ------------- | ------------ |
| 输入矩阵$x$   | 一个二维矩阵 |
| 输入卷积核$k$ | 一个二维矩阵 |
| 步长$stride$  | 始终为1      |
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
	1.在error周围填补上合适数目的0
	2.将kernel旋转180°
	3.将填补上0的误差和旋转之后的kernel进行步长为1的卷积，从而得到传递给下一层的误差new_error。
	
	# 更新参数
	1.将输入矩阵x和上一层传递来的误差矩阵error进行步长为1的卷积，得到kernel的更新梯度
	2.将上一层传递来的误差矩阵error所有元素求和，得到bias的更新梯度
	3.kernel := kernel - 学习率 * kernel的更新梯度
	4.bias := bias - 学习率 * bias的更新梯度
	
	# 返回误差，用以传递到下一层
	return new_error
```

