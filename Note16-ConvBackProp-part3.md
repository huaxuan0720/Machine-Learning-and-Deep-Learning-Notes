---
title: 步长stride为s的二维卷积方法的反向传播算法：一个十分极端的例子
date: 2019-05-24 12:32:45
tags: [卷积, 反向传播, 深度学习]
category: 深度学习
toc: true
thumbnail: gallery/DeepLearning.jpg
---

##### 前言  

&emsp;&emsp;在前面的文章中，介绍了二维平面上的卷积及其反向传播的算法，但是，步长为1和2毕竟都是两个比较小的数字，如果换成更大的数字，反向传播的方式是不是还适合呢？所以，我们考虑下面这个十分极端的例子，来验证反向传播算法的有效性。  

<!--more-->

##### 一、参数设置  
&emsp;&emsp;在之前的参数设置中，我们使用的输入矩阵都是5x5，在这篇文章中，我们使用10x10大小的矩阵，在卷积核方面，我们依然使用3x3大小的卷积核，步长stride方面，我们使用一个很大的数字7，padding方式依然设置为VALID。  

&emsp;&emsp;因此，我们的参数汇总如下：  

| 参数          | 设置                    |
| ------------- | ----------------------- |
| 输入矩阵$x$   | 一个二维矩阵，大小为10x10 |
| 输入卷积核$k$ | 一个二维矩阵，大小为3x3 |
| 步长$stride$  | 设置为7                 |
| padding       | VALID                   |
| 偏置项$b$     | 一个浮点数              |

&emsp;&emsp;和前面一样，我们定义卷积操作的符号为$conv$，我们可以将卷积表示为（需要注意的是这里步长选取为**7**）：
$$
x \; conv \; k + b = u
$$
&emsp;&emsp;展开之后，我们可以得到：  
$$
\begin{bmatrix}
x_{1, 1} & x_{1, 2} & x_{1, 3} &x_{1, 4} &x_{1, 5} & x_{1, 6} & x_{1, 7} & x_{1, 8} &x_{1, 9} &x_{1, 10}  \\
x_{2, 1} & x_{2, 2} & x_{2, 3} &x_{2, 4} &x_{2, 5} & x_{2, 6} & x_{2, 7} & x_{2, 8} &x_{2, 9} &x_{2, 10}  \\
x_{3, 1} & x_{3, 2} & x_{3, 3} &x_{3, 4} &x_{3, 5} & x_{3, 6} & x_{3, 7} & x_{3, 8} &x_{3, 9} &x_{3, 10}  \\
x_{4, 1} & x_{4, 2} & x_{4, 3} &x_{4, 4} &x_{4, 5} & x_{4, 6} & x_{4, 7} & x_{4, 8} &x_{4, 9} &x_{4, 10}  \\
x_{5, 1} & x_{5, 2} & x_{5, 3} &x_{5, 4} &x_{5, 5} & x_{5, 6} & x_{5, 7} & x_{5, 8} &x_{5, 9} &x_{5, 10}  \\
x_{6, 1} & x_{6, 2} & x_{6, 3} &x_{6, 4} &x_{6, 5} & x_{6, 6} & x_{6, 7} & x_{6, 8} &x_{6, 9} &x_{6, 10}  \\
x_{7, 1} & x_{7, 2} & x_{7, 3} &x_{7, 4} &x_{7, 5} & x_{7, 6} & x_{7, 7} & x_{7, 8} &x_{7, 9} &x_{7, 10}  \\
x_{8, 1} & x_{8, 2} & x_{8, 3} &x_{8, 4} &x_{8, 5} & x_{8, 6} & x_{8, 7} & x_{8, 8} &x_{8, 9} &x_{8, 10}  \\
x_{9, 1} & x_{9, 2} & x_{9, 3} &x_{9, 4} &x_{9, 5} & x_{9, 6} & x_{9, 7} & x_{9, 8} &x_{9, 9} &x_{9, 10}  \\
x_{10, 1} & x_{10, 2} & x_{10, 3} &x_{10, 4} &x_{10, 5} & x_{10, 6} & x_{10, 7} & x_{10, 8} &x_{10, 9} &x_{10, 10}  \\
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
将矩阵$u$进一步展开，我们有：  
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
x_{1, 8}k_{1, 1} + x_{1, 9}k_{1, 2} +x_{1, 10}k_{1, 3} + \\
x_{2, 8}k_{2, 1} + x_{2, 9}k_{2, 2} +x_{2, 10}k_{2, 3} + \\
x_{3, 8}k_{3, 1} + x_{3, 9}k_{3, 2} +x_{3, 10}k_{3, 3} + b \\
\end{matrix} \\ \\
\begin{matrix}
x_{8, 1}k_{1, 1} + x_{8, 2}k_{1, 2} +x_{8, 3}k_{1, 3} + \\
x_{9, 1}k_{2, 1} + x_{9, 2}k_{2, 2} +x_{9, 3}k_{2, 3} + \\
x_{10, 1}k_{3, 1} + x_{10, 2}k_{3, 2} +x_{10, 3}k_{3, 3} + b \\
\end{matrix} & \begin{matrix}
x_{8, 8}k_{1, 1} + x_{8, 9}k_{1, 2} +x_{8, 10}k_{1, 3} + \\
x_{9, 8}k_{2, 1} + x_{9, 9}k_{2, 2} +x_{9, 10}k_{2, 3} + \\
x_{10, 8}k_{3, 1} + x_{10, 9}k_{3, 2} +x_{10, 10}k_{3, 3} + b \\
\end{matrix} \\
\end{bmatrix}
$$

##### 二、误差传递  

&emsp;&emsp;和之前一样，为了方便计算，也为了方便观察，我们计算如下的表格，每一列表示的是一个特定的输出 $\partial u_{i, j}$，每一行表示的是一个特定的输入值$\partial x_{p, k}$，行与列相交的地方表示的就是二者相除的结果，表示的是输出对于输入的偏导数，即$\frac{\partial u_{i, j}}{\partial x_{p, k}}$。最后一列显示的是计算出的需要传递的误差的偏导数，具体计算方法和前面一样，在这里不再赘述：

|                       | $\partial u_{1, 1}$ | $\partial u_{1, 2}$ | $\partial u_{2, 1}$ | $\partial u_{2, 2}$ | $\frac{\partial L}{\partial x_{i, j}}$                       |
| --------------------- | :-----------------: | :-----------------: | :-----------------: | :-----------------: | :----------------------------------------------------------- |
| $\partial x_{1, 1}$   |     $k_{1, 1}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{1, 1}} = \delta_{1, 1} k_{1, 1}$ |
| $\partial x_{1, 2}$   |     $k_{1, 2}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{1, 2}} = \delta_{1, 1} k_{1, 2}$ |
| $\partial x_{1, 3}$   |     $k_{1, 3}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{1, 3}} = \delta_{1, 1} k_{1, 3}$ |
| $\partial x_{1, 8}$   |          0          |     $k_{1, 1}$      |          0          |          0          | $\frac{\partial L}{\partial x_{1, 8}} = \delta_{1, 2}k_{1, 1}$ |
| $\partial x_{1, 9}$   |          0          |     $k_{1, 2}$      |          0          |          0          | $\frac{\partial L}{\partial x_{1, 9}} = \delta_{1, 2}k_{1, 2}$ |
| $\partial x_{1, 10}$  |          0          |     $k_{1, 3}$      |          0          |          0          | $\frac{\partial L}{\partial x_{1, 10}} = \delta_{1, 2}k_{1, 3}$ |
| $\partial x_{2, 1}$   |     $k_{2, 1}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{2, 1}} = \delta_{1, 1} k_{2, 1}$ |
| $\partial x_{2, 2}$   |     $k_{2, 2}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{2, 2}} = \delta_{1, 1} k_{2, 2}$ |
| $\partial x_{2, 3}$   |     $k_{2, 3}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{2, 3}} = \delta_{1, 1} k_{2, 3}$ |
| $\partial x_{2, 8}$   |          0          |     $k_{2, 1}$      |          0          |          0          | $\frac{\partial L}{\partial x_{2, 8}} = \delta_{1, 2}k_{2, 1}$ |
| $\partial x_{2, 9}$   |          0          |     $k_{2, 2}$      |          0          |          0          | $\frac{\partial L}{\partial x_{2, 9}} = \delta_{1, 2}k_{2, 2}$ |
| $\partial x_{2, 10}$  |          0          |     $k_{2, 3}$      |          0          |          0          | $\frac{\partial L}{\partial x_{2, 10}} = \delta_{1, 2}k_{2, 3}$ |
| $\partial x_{3, 1}$   |     $k_{3, 1}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{3, 1}} = \delta_{1, 1} k_{3, 1}$ |
| $\partial x_{3, 2}$   |     $k_{3, 2}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{3, 2}} = \delta_{1, 1} k_{3, 2}$ |
| $\partial x_{3, 3}$   |     $k_{3, 3}$      |          0          |          0          |          0          | $\frac{\partial L}{\partial x_{3, 3}} = \delta_{1, 1} k_{3, 3}$ |
| $\partial x_{3, 8}$   |          0          |     $k_{3, 1}$      |          0          |          0          | $\frac{\partial L}{\partial x_{3, 8}} = \delta_{1, 2}k_{3, 1}$ |
| $\partial x_{3, 9}$   |          0          |     $k_{3, 2}$      |          0          |          0          | $\frac{\partial L}{\partial x_{3, 9}} = \delta_{1, 2}k_{3, 2}$ |
| $\partial x_{3, 10}$  |          0          |     $k_{3, 3}$      |          0          |          0          | $\frac{\partial L}{\partial x_{3, 10}} = \delta_{1, 2}k_{3, 3}$ |
| $\partial x_{8, 1}$   |          0          |          0          |     $k_{1, 1}$      |          0          | $\frac{\partial L}{\partial x_{8, 1}} = \delta_{2, 1} k_{1, 1}$ |
| $\partial x_{8, 2}$   |          0          |          0          |     $k_{1, 2}$      |          0          | $\frac{\partial L}{\partial x_{8, 2}} = \delta_{2, 1} k_{1, 2}$ |
| $\partial x_{8, 3}$   |          0          |          0          |     $k_{1, 3}$      |          0          | $\frac{\partial L}{\partial x_{8, 3}} = \delta_{2, 1} k_{1, 3}$ |
| $\partial x_{8, 8}$   |          0          |          0          |          0          |     $k_{1, 1}$      | $\frac{\partial L}{\partial x_{8, 8}} = \delta_{2, 2}k_{1, 1}$ |
| $\partial x_{8, 9}$   |          0          |          0          |          0          |     $k_{1, 2}$      | $\frac{\partial L}{\partial x_{8, 9}} = \delta_{2, 2}k_{1, 2}$ |
| $\partial x_{8, 10}$  |          0          |          0          |          0          |     $k_{1, 3}$      | $\frac{\partial L}{\partial x_{8, 10}} = \delta_{2, 2}k_{1, 3}$ |
| $\partial x_{9, 1}$   |          0          |          0          |     $k_{2, 1}$      |          0          | $\frac{\partial L}{\partial x_{9, 1}} = \delta_{2, 1} k_{2, 1}$ |
| $\partial x_{9, 2}$   |          0          |          0          |     $k_{2, 2}$      |          0          | $\frac{\partial L}{\partial x_{9, 2}} = \delta_{2, 1} k_{2, 2}$ |
| $\partial x_{9, 3}$   |          0          |          0          |     $k_{2, 3}$      |          0          | $\frac{\partial L}{\partial x_{9, 3}} = \delta_{2, 1} k_{2, 3}$ |
| $\partial x_{9, 8}$   |          0          |          0          |          0          |     $k_{2, 1}$      | $\frac{\partial L}{\partial x_{9, 8}} = \delta_{2, 2}k_{2, 1}$ |
| $\partial x_{9, 9}$   |          0          |          0          |          0          |     $k_{2, 2}$      | $\frac{\partial L}{\partial x_{9, 9}} = \delta_{2, 2}k_{2, 2}$ |
| $\partial x_{9, 10}$  |          0          |          0          |          0          |     $k_{2, 3}$      | $\frac{\partial L}{\partial x_{9, 10}} = \delta_{2, 2}k_{2, 3}$ |
| $\partial x_{10, 1}$  |          0          |          0          |     $k_{3, 1}$      |          0          | $\frac{\partial L}{\partial x_{10, 1}} = \delta_{2, 1} k_{3, 1}$ |
| $\partial x_{10, 2}$  |          0          |          0          |     $k_{3, 2}$      |          0          | $\frac{\partial L}{\partial x_{10, 2}} = \delta_{2, 1} k_{3, 2}$ |
| $\partial x_{10, 3}$  |          0          |          0          |     $k_{3, 3}$      |          0          | $\frac{\partial L}{\partial x_{10, 3}} = \delta_{2, 1} k_{3, 3}$ |
| $\partial x_{10, 8}$  |          0          |          0          |          0          |     $k_{3, 1}$      | $\frac{\partial L}{\partial x_{10, 8}} = \delta_{2, 2}k_{3, 1}$ |
| $\partial x_{10, 9}$  |          0          |          0          |          0          |     $k_{3, 2}$      | $\frac{\partial L}{\partial x_{10, 9}} = \delta_{2, 2}k_{3, 2}$ |
| $\partial x_{10, 10}$ |          0          |          0          |          0          |     $k_{3, 3}$      | $\frac{\partial L}{\partial x_{10, 10}} = \delta_{2, 2}k_{3, 3}$ |
| $else$                |          0          |          0          |          0          |          0          | 0                                                            |

&emsp;&emsp;可以看出，无论是何种卷积方式，数据都是十分有规律地进行分布。  

&emsp;&emsp;我们假设后面传递过来的误差是 $\delta$ ，即：
$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} \\
\delta_{2, 1} & \delta_{2, 2} \\
\end{bmatrix}
$$
&emsp;&emsp;其中，$\delta_{i, j} = \frac{\partial L}{\partial u_{i, j}}$，误差分别对应于每一个输出项。这里的$L$表示的是最后的Loss损失。我们的目的就是希望这个损失尽可能小。  

&emsp;&emsp;根据前面的方法，我们先要求应该传递给下一层的误差。所以第一步，我们先在接受来的误差矩阵中插入合适数目的0，由于这里前向卷积采用的步长stride是7，所以接收到误差矩阵中的每个元素之间应该插入（7 - 1 = 6）个0，即：
$$
\begin{bmatrix}
\delta_{1, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{1, 2} \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
\delta_{2, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{2, 2} \\
\end{bmatrix}
$$
&emsp;&emsp;接着，由于我们采用的卷积核的大小是3x3，所有，我们依然需要在上面矩阵的外围补上（3 - 1 = 2）层0，即：
$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{1, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{1, 2}  & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{2, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{2, 2}  & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$
&emsp;&emsp;下一步就是将正向卷积的卷积核旋转180°，即：
$$
\begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix}
$$
&emsp;&emsp;最后一步就是将上面的误差矩阵和旋转后的卷积核进行步长为1的卷积，即：
$$
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{1, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{1, 2}  & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \delta_{2, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{2, 2}  & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\end{bmatrix} \; conv(stride = 1)\; \begin{bmatrix}
k_{3, 3} & k_{3, 2} & k_{3, 1} \\
k_{2, 3} & k_{2, 2} & k_{2, 1} \\
k_{1, 3} & k_{1, 2} & k_{1, 1} \\
\end{bmatrix} = \\
\begin{bmatrix}
\delta_{1, 1} k_{1, 1} &  \delta_{1, 1} k_{1, 2} &  \delta_{1, 1} k_{1, 3} & 0 & 0 & 0 & 0 & \delta_{1, 2}k_{1, 1} & \delta_{1, 2}k_{1, 2} & \delta_{1, 2}k_{1, 3} \\
\delta_{1, 1} k_{2, 1} &  \delta_{1, 1} k_{2, 2} &  \delta_{1, 1} k_{2, 3} & 0 & 0 & 0 & 0 & \delta_{1, 2}k_{2, 1} & \delta_{1, 2}k_{2, 2} & \delta_{1, 2}k_{2, 3} \\
\delta_{1, 1} k_{3, 1} &  \delta_{1, 1} k_{3, 2} &  \delta_{1, 1} k_{3, 3} & 0 & 0 & 0 & 0 & \delta_{1, 2}k_{3, 1} & \delta_{1, 2}k_{3, 2} & \delta_{1, 2}k_{3, 3} \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
\delta_{2, 1} k_{1, 1} &  \delta_{2, 1} k_{1, 2} &  \delta_{2, 1} k_{1, 3} & 0 & 0 & 0 & 0 & \delta_{2, 2}k_{1, 1} & \delta_{2, 2}k_{1, 2} & \delta_{2, 2}k_{1, 3} \\
\delta_{2, 1} k_{2, 1} &  \delta_{2, 1} k_{2, 2} &  \delta_{2, 1} k_{2, 3} & 0 & 0 & 0 & 0 & \delta_{2, 2}k_{2, 1} & \delta_{2, 2}k_{2, 2} & \delta_{2, 2}k_{2, 3} \\
\delta_{2, 1} k_{3, 1} &  \delta_{2, 1} k_{3, 2} &  \delta_{2, 1} k_{3, 3} & 0 & 0 & 0 & 0 & \delta_{2, 2}k_{3, 1} & \delta_{2, 2}k_{3, 2} & \delta_{2, 2}k_{3, 3} \\
\end{bmatrix} = \\
\begin{bmatrix}
\frac{\partial L}{\partial x_{1, 1}} &
\frac{\partial L}{\partial x_{1, 2}} &
\frac{\partial L}{\partial x_{1, 3}} &
\frac{\partial L}{\partial x_{1, 4}} &
\frac{\partial L}{\partial x_{1, 5}} &
\frac{\partial L}{\partial x_{1, 6}} &
\frac{\partial L}{\partial x_{1, 7}} &
\frac{\partial L}{\partial x_{1, 8}} &
\frac{\partial L}{\partial x_{1, 9}} &
\frac{\partial L}{\partial x_{1, 10}} \\
\frac{\partial L}{\partial x_{2, 1}} &
\frac{\partial L}{\partial x_{2, 2}} &
\frac{\partial L}{\partial x_{2, 3}} &
\frac{\partial L}{\partial x_{2, 4}} &
\frac{\partial L}{\partial x_{2, 5}} &
\frac{\partial L}{\partial x_{2, 6}} &
\frac{\partial L}{\partial x_{2, 7}} &
\frac{\partial L}{\partial x_{2, 8}} &
\frac{\partial L}{\partial x_{2, 9}} &
\frac{\partial L}{\partial x_{2, 10}} \\
\frac{\partial L}{\partial x_{3, 1}} &
\frac{\partial L}{\partial x_{3, 2}} &
\frac{\partial L}{\partial x_{3, 3}} &
\frac{\partial L}{\partial x_{3, 4}} &
\frac{\partial L}{\partial x_{3, 5}} &
\frac{\partial L}{\partial x_{3, 6}} &
\frac{\partial L}{\partial x_{3, 7}} &
\frac{\partial L}{\partial x_{3, 8}} &
\frac{\partial L}{\partial x_{3, 9}} &
\frac{\partial L}{\partial x_{3, 10}} \\
\frac{\partial L}{\partial x_{4, 1}} &
\frac{\partial L}{\partial x_{4, 2}} &
\frac{\partial L}{\partial x_{4, 3}} &
\frac{\partial L}{\partial x_{4, 4}} &
\frac{\partial L}{\partial x_{4, 5}} &
\frac{\partial L}{\partial x_{4, 6}} &
\frac{\partial L}{\partial x_{4, 7}} &
\frac{\partial L}{\partial x_{4, 8}} &
\frac{\partial L}{\partial x_{4, 9}} &
\frac{\partial L}{\partial x_{4, 10}} \\
\frac{\partial L}{\partial x_{5, 1}} &
\frac{\partial L}{\partial x_{5, 2}} &
\frac{\partial L}{\partial x_{5, 3}} &
\frac{\partial L}{\partial x_{5, 4}} &
\frac{\partial L}{\partial x_{5, 5}} &
\frac{\partial L}{\partial x_{5, 6}} &
\frac{\partial L}{\partial x_{5, 7}} &
\frac{\partial L}{\partial x_{5, 8}} &
\frac{\partial L}{\partial x_{5, 9}} &
\frac{\partial L}{\partial x_{5, 10}} \\
\frac{\partial L}{\partial x_{6, 1}} &
\frac{\partial L}{\partial x_{6, 2}} &
\frac{\partial L}{\partial x_{6, 3}} &
\frac{\partial L}{\partial x_{6, 4}} &
\frac{\partial L}{\partial x_{6, 5}} &
\frac{\partial L}{\partial x_{6, 6}} &
\frac{\partial L}{\partial x_{6, 7}} &
\frac{\partial L}{\partial x_{6, 8}} &
\frac{\partial L}{\partial x_{6, 9}} &
\frac{\partial L}{\partial x_{6, 10}} \\
\frac{\partial L}{\partial x_{7, 1}} &
\frac{\partial L}{\partial x_{7, 2}} &
\frac{\partial L}{\partial x_{7, 3}} &
\frac{\partial L}{\partial x_{7, 4}} &
\frac{\partial L}{\partial x_{7, 5}} &
\frac{\partial L}{\partial x_{7, 6}} &
\frac{\partial L}{\partial x_{7, 7}} &
\frac{\partial L}{\partial x_{7, 8}} &
\frac{\partial L}{\partial x_{7, 9}} &
\frac{\partial L}{\partial x_{7, 10}} \\
\frac{\partial L}{\partial x_{8, 1}} &
\frac{\partial L}{\partial x_{8, 2}} &
\frac{\partial L}{\partial x_{8, 3}} &
\frac{\partial L}{\partial x_{8, 4}} &
\frac{\partial L}{\partial x_{8, 5}} &
\frac{\partial L}{\partial x_{8, 6}} &
\frac{\partial L}{\partial x_{8, 7}} &
\frac{\partial L}{\partial x_{8, 8}} &
\frac{\partial L}{\partial x_{8, 9}} &
\frac{\partial L}{\partial x_{8, 10}} \\
\frac{\partial L}{\partial x_{9, 1}} &
\frac{\partial L}{\partial x_{9, 2}} &
\frac{\partial L}{\partial x_{9, 3}} &
\frac{\partial L}{\partial x_{9, 4}} &
\frac{\partial L}{\partial x_{9, 5}} &
\frac{\partial L}{\partial x_{9, 6}} &
\frac{\partial L}{\partial x_{9, 7}} &
\frac{\partial L}{\partial x_{9, 8}} &
\frac{\partial L}{\partial x_{9, 9}} &
\frac{\partial L}{\partial x_{9, 10}} \\
\frac{\partial L}{\partial x_{10, 1}} &
\frac{\partial L}{\partial x_{10, 2}} &
\frac{\partial L}{\partial x_{10, 3}} &
\frac{\partial L}{\partial x_{10, 4}} &
\frac{\partial L}{\partial x_{10, 5}} &
\frac{\partial L}{\partial x_{10, 6}} &
\frac{\partial L}{\partial x_{10, 7}} &
\frac{\partial L}{\partial x_{10, 8}} &
\frac{\partial L}{\partial x_{10, 9}} &
\frac{\partial L}{\partial x_{10, 10}} \\
\end{bmatrix}
$$

&emsp;&emsp;经过上面的计算，在误差传递上，我们的算法可以正确运行，即使步长stride是一个任意的数字。接下来我们来验证更新梯度的计算。  

##### 三、更新梯度  

&emsp;&emsp;和前面的定义一样，假设我们在这一阶段接收到的后方传递过来的误差为$\delta$， ，即：
$$
\delta = 
\begin{bmatrix}
\delta_{1, 1} & \delta_{1, 2} \\
\delta_{2, 1} & \delta_{2, 2} \\
\end{bmatrix}
$$
&emsp;&emsp;那么根据偏导数求解的链式法则，我们可以计算出所有的需要的偏导数，这里的计算过程和前面的计算过程是一样的，这里不再赘述。汇总如下：
$$
\frac{\partial L}{\partial k_{1, 1}} = x_{1, 1}\delta_{1, 1} + x_{1, 8}\delta_{1, 2} + x_{8, 1}\delta_{2, 1} + x_{8, 8}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{1, 2}} = x_{1, 2}\delta_{1, 1} + x_{1, 9}\delta_{1, 2} + x_{8, 2}\delta_{2, 1} + x_{8, 9}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{1, 3}} = x_{1, 3}\delta_{1, 1} + x_{1, 10}\delta_{1, 2} + x_{8, 3}\delta_{2, 1} + x_{8, 10}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{2, 1}} = x_{2, 1}\delta_{1, 1} + x_{2, 8}\delta_{1, 2} + x_{9, 1}\delta_{2, 1} + x_{9, 8}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{2, 2}} = x_{2, 2}\delta_{1, 1} + x_{2, 9}\delta_{1, 2} + x_{9, 2}\delta_{2, 1} + x_{9, 9}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{2, 3}} = x_{2, 3}\delta_{1, 1} + x_{2, 10}\delta_{1, 2} + x_{9, 3}\delta_{2, 1} + x_{9, 10}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{3, 1}} = x_{3, 1}\delta_{1, 1} + x_{3, 8}\delta_{1, 2} + x_{10, 1}\delta_{2, 1} + x_{10, 8}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{3, 2}} = x_{3, 2}\delta_{1, 1} + x_{3, 9}\delta_{1, 2} + x_{10, 2}\delta_{2, 1} + x_{10, 9}\delta_{2, 2}
$$
$$
\frac{\partial L}{\partial k_{3, 3}} = x_{3, 3}\delta_{1, 1} + x_{3, 10}\delta_{1, 2} + x_{10, 3}\delta_{2, 1} + x_{10, 10}\delta_{2, 2}
$$

$$
\frac{\partial L}{\partial b} = \delta_{1, 1} + \delta_{1, 2} + \delta_{2, 1} + \delta_{2, 2}
$$

&emsp;&emsp;按照之前的算法，由于正向卷积中的步长stride为7，因此，在计算更新梯度的过程中，我们依然需要在接收到的误差矩阵的每两个相邻的元素之间插入（7 - 1 = 6）个0，即：
$$
\begin{bmatrix}
\delta_{1, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{1, 2} \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \\
\delta_{2, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{2, 2} \\
\end{bmatrix}
$$

&emsp;&emsp;接着我们拿输入矩阵$x$和上面的矩阵进行步长为1的卷积，则可以得到卷积核参数的更新梯度。即：  
$$
\begin{bmatrix}
\frac{\partial L}{\partial k_{1, 1}} & \frac{\partial L}{\partial k_{1, 2}} & \frac{\partial L}{\partial k_{1, 3}} \\
\frac{\partial L}{\partial k_{2, 1}} & \frac{\partial L}{\partial k_{2, 2}} & \frac{\partial L}{\partial k_{2, 3}} \\
\frac{\partial L}{\partial k_{3, 1}} & \frac{\partial L}{\partial k_{3, 2}} & \frac{\partial L}{\partial k_{3, 3}} \\
\end{bmatrix} = \\ 
\begin{bmatrix}
x_{1, 1} & x_{1, 2} & x_{1, 3} &x_{1, 4} &x_{1, 5} & x_{1, 6} & x_{1, 7} & x_{1, 8} &x_{1, 9} &x_{1, 10}  \\
x_{2, 1} & x_{2, 2} & x_{2, 3} &x_{2, 4} &x_{2, 5} & x_{2, 6} & x_{2, 7} & x_{2, 8} &x_{2, 9} &x_{2, 10}  \\
x_{3, 1} & x_{3, 2} & x_{3, 3} &x_{3, 4} &x_{3, 5} & x_{3, 6} & x_{3, 7} & x_{3, 8} &x_{3, 9} &x_{3, 10}  \\
x_{4, 1} & x_{4, 2} & x_{4, 3} &x_{4, 4} &x_{4, 5} & x_{4, 6} & x_{4, 7} & x_{4, 8} &x_{4, 9} &x_{4, 10}  \\
x_{5, 1} & x_{5, 2} & x_{5, 3} &x_{5, 4} &x_{5, 5} & x_{5, 6} & x_{5, 7} & x_{5, 8} &x_{5, 9} &x_{5, 10}  \\
x_{6, 1} & x_{6, 2} & x_{6, 3} &x_{6, 4} &x_{6, 5} & x_{6, 6} & x_{6, 7} & x_{6, 8} &x_{6, 9} &x_{6, 10}  \\
x_{7, 1} & x_{7, 2} & x_{7, 3} &x_{7, 4} &x_{7, 5} & x_{7, 6} & x_{7, 7} & x_{7, 8} &x_{7, 9} &x_{7, 10}  \\
x_{8, 1} & x_{8, 2} & x_{8, 3} &x_{8, 4} &x_{8, 5} & x_{8, 6} & x_{8, 7} & x_{8, 8} &x_{8, 9} &x_{8, 10}  \\
x_{9, 1} & x_{9, 2} & x_{9, 3} &x_{9, 4} &x_{9, 5} & x_{9, 6} & x_{9, 7} & x_{9, 8} &x_{9, 9} &x_{9, 10}  \\
x_{10, 1} & x_{10, 2} & x_{10, 3} &x_{10, 4} &x_{10, 5} & x_{10, 6} & x_{10, 7} & x_{10, 8} &x_{10, 9} &x_{10, 10}  \\
\end{bmatrix} \; conv(stride = 1)\; \begin{bmatrix}
\delta_{1, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{1, 2} \\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
\delta_{2, 1} & 0 & 0 & 0 & 0 & 0 & 0 & \delta_{2, 2} \\
\end{bmatrix}
$$

&emsp;&emsp;经过计算，两者的结果是相同的，这也就验证了我们的算法在一些比较极端的情况下也是正确的。  

##### 四、总结  

&emsp;&emsp;经过一个比较极端的卷积实例的讲解，我们验证了我们算法的正确性，而下一步就是用代码实现二维平面上的卷积及其反向传播算法。

