---
layout: post
title:  "Classification using binary logistic regression"
date:   2018-09-14 16:57:51
---

Logistic Regression is an extension of simple linear regression  and it's used when the dependent variable(target) is categorical. There's three types of logistic regression: binary, multinomial and ordinal logistic regression but this article is only about binary version of this algorithm.


The logistic regression model is simply a non-linear transformation of the linear regression and its distribution is an S-shaped distribution function (cumulative density function) which is similar to the standard
normal distribution and constrains the estimated probabilities to lie between 0
and 1.

![s-shaped distribution](http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png)
[source][s-shaped]

#### how it works

This algorithm estimates the relationship between the dependent variable, what we want to predict, and the one or more independent variables or features, by estimating probabilities using logistic function.


#### few practical cases

* You could use binomial logistic regression to understand whether exam performance can be predicted based on revision time, test anxiety and lecture attendance (i.e., where the dependent variable is "exam performance", measured on a dichotomous scale – "passed" or "failed" – and you have three independent variables: "revision time", "test anxiety" and "lecture attendance").

* ..or in order to determine if the email is spam (1) or not (0)

#### more practice


[s-shaped]: http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png
