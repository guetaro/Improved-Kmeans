# Improved-Kmeans

This is my final project in the Unsupervised learning class, by Dr Ofir lindenboum in Bar Ilan university.

The project's goal is to find new ways to improve the original K means algorithm regarding running time and accuracy.

## Description

In the project I implemented the original K means algorithm and the improved K means algorithm as presented in https://ieeexplore.ieee.org/abstract/document/5453745?casa_token=0KVvU3e_IC4AAAAA:A5JN1Xln54pqKxf3m4BNWePZ66ojye2w92PUM7WlZUfagoDVIHGkYGBqJUedB_QyA6cn5DepBw. 

Moreover, I implemented and tested both centroid initialization algorithms:
The min-Max algorithm as presented in: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=329844&casa_token=Oh6VLSJx_EwAAAAA:q_bGltIDyaFm29oKuGaQAZCk6-VgFQwe8aiWQ-I2UUhOAbLmAU3Zb5msPQv_8uIgPJUJgBjVsg
The Max var algorithm as presented in:
https://www.ire.pw.edu.pl/~arturp/Dydaktyka/PPO/pomoce/cluster_gaussian_mixture.pdf

Finally, I presented a usage of the PCA algorithm for dimensionality reduction to the K means algorithm.

I tested the methods on several well-Known datasets, and it seems the PCA algorithm presented the best results when it comes to time consumption.

## Datasets
Mnist , Letter , Iris

## packages 
keras , sklearn , numpy , matplotlib , time , scipy , pandas , sklearn
