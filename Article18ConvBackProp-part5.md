---
title: 多通道卷积以及激活函数
date: 2019-05-27 15:09:30
tags: [卷积, 反向传播, 深度学习]
category: 深度学习
toc: true
thumbnail: gallery/DeepLearning.jpg
---

##### 前言  

&emsp;&emsp;前面讲了很多二维平面上的卷积，甚至用代码实现了一个简单的两层二维卷积网络，但是在实际的情况下，我们使用的更多的是三维矩阵，即矩阵的$shape$往往是$[height, width, channels]$。在这种情况下，我们的卷积核就会多出一个参数来和通道$channels$参数进行匹配，即，这个时候，我们的卷积核的$shape$会变成$[kernel\_height, kernel\_width, channels]$。所以接下来就是要弄清楚在这种多通道的情况下，卷积是如何进行反向传播的。  

<!--more-->

##### 一、参数定义  

&emsp;&emsp;由于带有多个通道属性，因此，我们可以将每个通道的数据都视为一个二维的矩阵，卷积核也按照通道的数目拆分成多个二维的卷积核数据，因此，当我们分别对这些二维矩阵进行卷积之后，在将所有的结果相加即可得到最后的结果。这就是带有通道数据的卷积方式的本质。反过来，我们也利用这种本质，来进行反向传播。  

&emsp;&emsp;在本文的模型中，我们假定平面上的卷积为$plane\_conv(x, kernel, stride)$，再定义带有数据通道的卷积函数为$conv(x, kernel, stride)$，其中，$x$和$kernel$均为三维矩阵，格式分别为：$x.shape: [height, width, channels]$，$kernel.shape: [kernel\_height, kernel\_width, channels]$，则我们有：
$$
conv(x, kernel, stride) = \sum_{i = 0}^{channels} plane\_conv(x_{i}, kernel_{i}, stride)
$$
&emsp;&emsp;上面的公式的含义就是多通道卷积产生的结果是将各个通道进行分开，然后在每个通道上分别进行二维平面上的卷积，最后将这些二维的卷积结果相加得到的，而事实上，这本身就是多通道卷积的本质含义。  

&emsp;&emsp;所以当我们对其中的每一通道的$kernel_{i}$和$x_i$求导时，我们有：  
$$
\frac{\partial conv(x, kernel, stride)}{\partial x_i} = \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial x_i}
$$

$$
\frac{\partial conv(x, kernel, stride)}{\partial kernel_i} = \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial kernel_i}
$$

&emsp;&emsp;现在，我们假设由上层传来的误差为$\delta = \frac{\partial L}{\partial conv(x, kernel, stride)}$，那么我们在上式的两边同时乘以误差$\delta$，根据求导的链式法则，我们有：  
$$
\frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial conv(x, kernel, stride)}{\partial x_i} =  \frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial x_i}
$$

$$
\frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial conv(x, kernel, stride)}{\partial kernel_i} = \frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial kernel_i}
$$

&emsp;&emsp;化简之后，我们有：  
$$
\frac{\partial L}{\partial x_i} =  \frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial x_i}
$$

$$
\frac{\partial L}{\partial kernel_i} = \frac{\partial L}{\partial conv(x, kernel, stride)} \cdot \frac{\partial plane\_conv(x_{i}, kernel_{i}, stride)}{\partial kernel_i}
$$

&emsp;&emsp;上述的两个等式的左边就是我们需要计算的每一个通道的偏导数，即向前传递的误差的偏导数和需要用来更新参数的偏导数，等式右边则变成了接收到的误差矩阵和每个通道上的二维卷积求导数，因此等式右边就变成了我们已经知道的在二维平面上进行反向传播的偏导数求解了。所以我们将问题由三维空间分解到了二维空间，而二维空间上的卷积及其反向传播我们在之前已经花了较多篇幅进行讲解，所以这里不再赘述了。  

&emsp;&emsp;需要注意的是，上面的推导过程都仅限于一个卷积核，若有多个卷积核则需要在每一个卷积核上利用这个方法进行偏导数的求解。