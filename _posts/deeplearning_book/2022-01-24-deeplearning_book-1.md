---
title: "[심층학습 책 리뷰] 심층학습(Deep learning) 책9장 Convolutional Networks 리뷰"
excerpt: "9장 CNN에서 기본적인 원리와 내용에 대해 알아본다."

date: 2022-01-24
categories:
 - deeplearning_book
tags:
  - deeplearning
  - cnn
  - vision
  - book_review
  - DL
layout: jupyter
search: true

# 목차
toc: true  
toc_sticky: true 

use_math: true
---

## 0. Introduction

✍🏻 이번 포스팅에서는 이안 굿펠로 외의 Deep Learning 책 9장 **Convolutional Networks** 내용에 대해 살펴본다.

- Book : [Deep Learning](https://www.deeplearningbook.org/)
          (2015 / Goodfellow, Ian , Bengio, Yoshua/ Courville, Aaron)


9장에서는 convolution이 무엇인지 설명하고 convolution의 동기, 즉 신경망에서 convolution을 사용하는 이유를 제시한다. 그리고 convolution와 관련된 다양한 용어를 설명하고 그에 대한 내용을 추가적으로 이야기한다. 

---

목차는 다음과 같이 구성되어 있다.
- 9-1. Convolution Operation(합성곱 연산)
- 9-2. Motivation(동기)
- 9-3 Pooling(풀링)
- 9-4. Convolution and Pooling as an Infinitely Strong Prior(무한히 강한 사전분포로서의 합성곱과 풀링)
- 9-5. Variants of the Basic Convolution Function(기본 합성곱 함수의 여러 변형)
- 9-6. Structured Outputs(구조적 출력)
- 9-7. Data types(자료 형식)
- 9-8. Efficient Convolution Algorithms(효율적인 합성곱 알고리즘)
- 9-9. Random or Unsupervised Features(무작위 특징 또는 비지도 학습 특징 학습)
- 9-10. The Neuroscientific Basic for Convolutional Networks
- 9-11. Convolutional Networks and the History of Deep Learning(합성곱 신경망으로 본 심층 학습의 역사)

---

## 1. Convolution Operation(합성곱 연산)

Convolution Network은 **Convolution**이라는 수학 연산을 사용하기 때문에 붙은 이름이다. <br>
<br>

**Convolution이란 무엇일까?**
- 