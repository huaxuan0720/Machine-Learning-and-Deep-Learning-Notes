---
title: Adaboost算法
date: 2019-05-10 11:41:51
tags: [机器学习, AdaBoost]
categories: 机器学习
toc: true
thumbnail: gallery/MachineLearning.jpg
---



##### 前言  

&emsp;&emsp;提升方法（boosting）是一种常用的机器学习方法，应用十分广泛，而且效果非常好，近几年的很多比赛的优胜选手都或多或少使用了提升方法用以提高自己的成绩。

&emsp;&emsp;提升方法的本质是通过对每一个训练样本赋予一个权重，并通过改变这些样本的权重，来学习多个分类器，并按照一定的算法将这些分类器组合在一起，通常是线性组合，因为单个分类器往往效果有限，因此组合多个分类器往往会提高模型的性能。 

<!-- more --> 

##### 一、提升方法简介  

&emsp;&emsp;提升方法（boosting）实际上是集成学习方法的一种，其基于这样的一种朴素思想：对于一个复杂的任务来说，将多个“专家”（这里的“专家”本意是指各种机器学习模型，可以较好地满足实际问题的需要）的意见进行适当的整合，进而得出最后的综合的判断，比其中任何一个单一的“专家”给出的判断要好。实际上，也就是“三个臭皮匠顶过诸葛亮”的意思。因此，boousting的本意就是寻找到合适的“臭皮匠”。  

&emsp;&emsp;在实际的数据处理的过程中，我们往往可以很容易地发现各种各样的弱机器学习模型，这些模型仅仅比随机猜测好一些，但是还远远不能满足实际作业的精度要求。但是要寻找到一个单一的十分强大的机器学习模型往往会十分困难，虽然可以很好的满足要求，但是寻找这样的模型并不容易。不过好在，我们可以通过整合之前发现的弱机器学习模型，来进行综合考虑，从而形成一个可以媲美单一的强大的机器学习模型。这些弱机器学习模型往往被称之为“弱学习方法”，强机器学习模型往往被称之为“强学习方法”。  

&emsp;&emsp;



##### 二、AdaBoost算法  

###### 	1、AdaBoost算法的过程  

&emsp;&emsp;

###### 	2、AdaBoost的使用