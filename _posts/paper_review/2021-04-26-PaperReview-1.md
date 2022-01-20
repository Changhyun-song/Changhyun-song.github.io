---
title: "[Paper Review] Improving Calibration for Long-Tailed Recognition (ICLR 2021) 논문 리뷰"
excerpt: "Calibration을 향상시킬 수 있는 다양한 방법에 대해 소개한다."

date: 2021-04-26
categories:
 - Paper_Review
tags:
  - paper_review
  - calibration
  - vision
  - deeplearning
layout: jupyter
search: true

# 목차
toc: true  
toc_sticky: true 

use_math: true
---

> ✍🏻 이번 포스팅에서는 적은 이미지로 학습가능한 **Improving Calibration for Long-Tailed Recognition** 논문 리뷰를 꼼꼼하게 !!

- Paper : [Improving Calibration for Long-Tailed Recognition](https://arxiv.org/abs/2104.00466) (arxiv 2021 /Zhisheng Zhong, Jiequan Cui, Shu Liu, Jiaya Jia)

---

## 1. Abstract

본 논문에서는 ***Deep Neural Networks***가 ***Training Datasets*이 심한 *Class-Imbalance*가 있을 경우 성능이 저하될 수 있다**고 말하고 있다.<br><br> *Two-stage Method*를 통해 *Representation Learning*과 *Classifier learning* 성능을 향상시키긴 했지만 여전히 ***Miscalibration*이 발생**한다.<br><br> 이를 해결하기 위해 본 논문에서는 2가지 방법을 제안한다.
<br><br>
***"Motivated by the fact that predicted probability distributions of classes are highly related to the numbers of class instances, we propose label-aware smoothing to deal with different degrees of over-confidence for classes and improve classifier learning.<br>For dataset bias between these two stages due to different samplers, we further propose shifted batch normalization in the decoupling framework."***

---

## 2. Introduction
많이 쓰이는 Open Dataset 같은 경우에는 일반적으로 각각의 Object, Class의 Instance 수와 관련해서 인위적으로 균형을 이루고 있다. <br><br>하지만 실제 사용되는 일반적인 데이터셋은 각각의 Class의 Instance 수가 심각하게 불균형한 ***Long-tailed Distribution***을 보여주고 있다. Long-tailed Distribution애 대해 CNN을 학습시킬 때 성능이 크게 떨어진다.
* * *
여기서 ***Long-tailed Distribution*** 이란?
<br><br>
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-1.png?raw=1" width = "800" ></p>

쉽게 말하면 **클래스가 가지고 있는 데이터 양의 차이가 큰 것**을 말한다. <br><br>예를 들면 병원에서 질병이 있는 사람과 질병이 없는 사람의 데이터를 모아야 한다고 했을 때 일반적으로 질병이 있는 사람의 데이터가 질병이 없는 사람의 데이터 수에 비해 현저하게 적다. <br><br>물론 병원 데이터 뿐 아니라 현실 데이터에서는 대부분 클래스 불균형 문제를 가지고 있다. <br><br>이러한 클래스 불균형으로 인해 특정 클래스의 Instance가 너무 높고 반대로 다른 클래스의 경우는 매우 낮기 때문에 마치 긴 꼬리 모양과 같이 생긴 것을 ***Long-tailed Distribution***이라고 말한다.
<br>
* * *
다시 논문으로 돌아오면 최근에는 *Two-Stage Approach*를 통해서 성능이 *One-stage Method*와 비교했을 때 상당히 개선되었다.<br><br> *Two-Stage Approach*에서<br> ***Deffered Re-sampling(DRS)***과 ***Deffered Re-weighting(DRW)***방법이 있다.
<br>
- 1. 일반적인 방법으로 불균형되어 있는 Dataset을 CNN Model로 학습시킨다.
- 2. DRS로 클래스 균형 리샘플링을 사용하여 데이터 세트에서 CNN을 조정한다.
- 3. DRW로 클래스에 다른 weight를 할당함으로써 CNN을 조정한다.

본 논문에서 참고한 2가지 논문과 링크는 아래에 첨부하겠습니다.
<br><br>
**_bbn: bilateral-branch network with cumulative learning for long-tailed visual recognition, CVPR 2020_** [Link](https://arxiv.org/abs/1912.02413 "논문 링크")
<br><br>
**_Decoupling Representation and Classifier for Long-Tailed Recognition, ICLR 2020_** [Link](https://arxiv.org/abs/1910.09217 "논문 링크")
<br><br>
첫 번째 논문에서는 ***Bilateral-Branch Network(BBN-Model)***을 제안한다.<br> 이 모델은 Representation learning과 classifier learning을 따로 수행하는 형태의 학습 방법을 의미한다. 
<br><br>
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-2.png?raw=1" width = "800" ></p>
<br><br>
위 모델은 2가지의 branch로 이루어져 있다.<br>

### 1. Coventional learning branch
- Representation learning
- 원래 ***Long-tail distribution pattern*을 그대로 학습하는 용도로 사용**된다.
- ***Typical uniform sampler*** 사용<br><br>
이 때 ***Typical uniform sampler***는 클래스 불균형이 있는 Dataset을 그대로 sampling 하는 것을 의미한다. 따라서 이 때 학습할 때 data가 많은 class, 즉 Head 쪽 Data가 학습이 더 많이 진행되게 되고 결과적으로 feature의 학습인 representation learning이 더 잘 되게 만든다.
<br><br>

### 2. Re-balancing branch
- Classifier learning
- Coventional learning branch와 달리 Tail 쪽 Data를 조금 더 많이 sampling한다. 
- Tail class에 대한 classification accuracy를 상승시키기 위한 것이다.

<br><br>
이 논문으로부터 얻을 수 있는 정보는 *Original Data*로 부터 *Feature learning*의 장점을 얻기 위해<br><br> **Conventional branch를 통해서 Original distribution에 대해 학습**을 진행한다.<br><br>
이전 실험에서 *Representation learning*을 한 이후에 *Classifier learning*을 *RW, RS* 형태로 진행한 것처럼 *Alpha* 값을 조정하여 처음에는 *uniform sampler*로부터 학습을 시작하고,<br> **이것으로부터 feature가 잘 학습된 Backbone 네트워크로부터 RS/RW 효과를 내는 Re-balancing branch로 부터 학습을 늘린다.** <br><br>
최종적으로<br> ***Conventional learning branch***는 ***Majority Class***에 preference를 더 가지도록, <br>***Re-balancing branch***는 ***Minority Class***에 preference를 더 가지도록 학습이 된다.

따라서 Mix-up을 했을 때, 즉 두 가지 결과를 합쳤을 때 이런 Weight가 Balance한 형태로 가장 잘 맞춰지게 된다.<br><br>
두 번째 논문에서는 ***Two-stage decoupling Model***을 제안한다.<br> 이 모델은 ***classifier re-training(cRT)***와 ***Learnable weight scaling(LWS)***가 있다.

#### 1. classifier re-training(cRT)
**Representation learning 부분을 고정**시키고 *Classifier*만 *Class Balanced* 형식으로 다시 학습시키는 방법을 이야기 한다.

#### 2. Learnable weight scaling(LWS)
*Scaling* 하는 정도는 학습을 통해서 얻는 방법을 의미한다.
<br><br>
***Confidence Calibration***<br>
*Calibration*이란 모형의 출력값이 실제 *Calibrated Confidence*를 반영하도록 만드는 것을 말한다. <br><br>
예를 들어 X의 Y1에 대한 모형의 출력이 0.8이 나왔을 때, 80% 확률로 Y1일 것이라는 의미를 갖도록 만드는 것이다. <br>일반적으로 현대 딥러닝같은 경우에는 ***Overconfident*** 성격을 띄고 있다. <br><br>예시로 아래 그림을 보면 1998년에 제시된 *LeNet*의 경우 모형의 출력이 0~1 사이에 균일하게 분포되어 있지만, *ResNet*의 경우 1근처에 집중되어 있는 것을 확인할 수 있다. <br><br>그 결과로 아래 그림을 보게 되면 *ResNet*의 경우 *Confidence*와 *Accuracy*가 많이 어긋나는 것을 확인할 수 있다. <br>
**모형의 출력이 실제 *Calibrated Confidence*를 반영한다면 *Confidence*와 *Accuracy*는 일치해야 한다.**
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-3.png?raw=1" width = "800" ></p>
**모형의 예측값이 실제 확률을 반영한다는 의미를 가진 *Calibration*이 중요한 이유**<br><br>
실제 딥러닝이 응용될 때, 의사결정 프로세스중 하나의 구성요소가 될 경우가 많다. <br><br>의학적 진단을 예로 들자면, 딥러닝을 전적으로 신뢰해서 모든 판단을 딥러닝에게 맡기는 의사결정이 이루어지는 경우는 적고, 딥러닝 모델의 *Confidence*가 낮은 경우에만 사람이 보는 방식으로 사람이 할 일의 일부를 딥러닝이 하게 되는 구조가 대부분입니다. <br><br>이 경우 *Confidence*가 낮은 것만 사람이 재확인하는 방식이 가능한데 이러한 의사결정이 가능하기 위해서는 모형의 *Confidence*를 보는 것이 필요하고 **이 *Confidence*가 *Calibrated Confidence*이여야 신뢰할 수 있는 값**이라고 할 수 있다.<br><br>

네트워크의 *Calibration*을 측정하는 방법으로 ***Expected Calibration Error(ECE)***를 사용한다. <br><br> ***ECE***는 ***Confidence*와 실제 *Accuracy*의 *Distribution*의 차이를 측정하는 방법**이다. 예측값을 균등하게 M묶음으로 나눈 뒤에 *Accuracy*와 *Confidence*차이의 평균을 계산하는 방법이다. 여기서 n은 샘플의 개수다.
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-4.png?raw=1" width = "800" ></p>

아래 그림을 통해 클래스의 불균형 구성 비율 때문에 *Long-tailed Datasets*에서 훈련된 네트워크가 ***Miscalibrated***하고 ***Over-Confident***하다는 것을 보여준다. <br>
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-5.png?raw=1" width = "800" ></p>

- 원본 CIFAR-100 Dataset + CE(Cross Entropy)
- Long-tailed Datasets
- Long-tailed Datasets + cRT
- Long-tailed Datasets + LWS

위 결과값을 통해 *Long-tailed Datasets*를 훈련한 *Network*가 일반적으로 ECE가 높다는 것을 확인할 수 있다. 또한 *cRT, LWS* 에서도 마찬가지로 *Over-Confidence*를 확인할 수 있다. <br>위 현상은 다른 *Long-tailed Datasets*에서도 존재한다.
<br><br>
또 다른 문제는  *Two-stage Decoupling*이 *Dataset bias* 또는 *Domain shift*를 무시한다는 것이다.<br><br>
이 문제는 1단계에서 *Instanced balanced Dataset*에 대해 먼저 훈련하고 2단계에서 모델이 *Class-balanced dataset*에서 훈련했을 때 <br>
***Distribution of the dataset by different sampling ways*가 일치하지 않는다는 것이다.**<br><br>
따라서 *Dataset bias* 문제를 해결하기 위해서 Batch normalization에 초점을 둔다. <br><br>위 문제들을 모두 해결하기 위해서 논문에서는 ***Mixup Shifted Label-Aware Smoothing model(MiSLAS)***를 제안한다.<br>
- *Long-tailed Dataset*에 대해 훈련된 *Model*이 *Balanced Dataset*에 대해 훈련된 *Model* 보다 훨씬 ***Miscalibrated and Over-confident*** 
<br>(이는 2단계 모델 역시 같은 문제점 발생)
- *mixup*은 *representation learning*에는 긍정적인 영향을 주고 *Over-confidence*를 줄일 수 있지만 *Classifier learning*에서는 부정적 영향을 줄 수 있다.<br>**따라서 *Classifier learning*과 *Calibration*을 향상시키기 위해 *Label-aware smoothing*을 제안한다.**<br><br>***Label-aware smoothing***이란? <br>***-->handle different degrees of over- confidence for classes***
- *Dataset bias or Domain shift*를 *Decoupling Framework*에서 해결하기 위해 성능을 개선할 수 있는 ***Shift learning on the batch normalization layer***를 제안한다.<br>
- *Long-tailed Dataset* 여러 개에서 MiSLAS를 검증하고 실험 결과를 보여준다.

---

## 3. Related Work

***Re-sampling and Re-weighting***<br>
1. *Re-samling*
<br>
- ***Over-sampling the Tail-class images***<br>
  - *Over-sampling*은 대규모 데이터셋에 정기적으로 유용
  - 소규모 데이터셋에서 ***Tail Class*에 대한 *Over-fitting* 발생**<br>
- ***Under-sampling the Head-class images***<br>
  - 데이터의 많은 부분을 폐기하므로 ***Deep model*의 일반화 능력이 저하**<br>

2. *Re-weighting*
<br>
- *Class & Instance*에 서로 다른 *weight*를 할당
- *Vanilla Re-weighting Method*
  - 클래스 샘플 수에 **역비례**하여 ***Class weight*를 제공**
- 대규모 데이터의 경우 학습시키는 동안 *Deep Model*을 최적화하기 어려움
  - 유효 숫자를 사용하여 *Class weight*를 사용하여 위 문제 해결
  - 각 인스턴스의 *weight*를 적응적으로 다시 매김 <br>(ex.***Focal loss*** -> 잘 분류된 예제에는 작은 *weight*, 분류하기 어려운 일부 예제에는 큰 *weight*를 부여하여 학습을 어려운 예제에 집중시킴)

***Confidence calibration and regularization***<br>
- *Calibrated confidence*는 *Classification model*에서 중요<br>
  - *Model capacity, Normalization, Regularization -> Network Calibration*에 큰 영향을 미치는 것을 확인
- *Mixup*<br>*Interpolation of input and labels*으로 훈련하는 *Regularization* 기법
    - *manifold mixup, Cut- Mix, Remix*
    - *Mixup*으로 학습된 *CNN* -> ***Better calibrated*** 
- *Label smoothing*<br>*another Regularization* 기법 <br>-->**Over-confident**를 줄이도록 *Model*을 만든다.
    - *compute loss upon a soft version of labels*
    - *relieve Over-fitting and increase Calibration and Reliability*

***Mixup***은 학습을 진행할 때 랜덤하게 두 개의 샘플 (x(i),y(i)), (x(j),y(j))를 뽑아서 (x_dot,y_dot)을 만들어 학습에 사용하는 것을 말한다.<br>
아래 그림은 ***Mixup***에 대한 예시를 보여준다. *lambda*는 보통 0.5가 아닌 한 쪽 데이터에 치우치도록 0.1 혹은 0.2정도를 준다.
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-8.png?raw=1" width = "800" ></p>
<br><br>
***Label smoothing***은 일반화 성능을 높이기 위해 사용하는 기법이다.<br> *one-hot encoding*처럼 정답 레이블에 1 아닌 레이블에 0을 부여하는 것이 아니라 <br>정답 레이블이 아닌 레이블에도 약간의 레이블 값을 넣어주는 것을 의미한다.<br>
아래 그림은 *Label smoothing*에 대한 예시이다.<br><br>
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-9.png?raw=1" width = "800" ></p>

***Two-stage methods***<br>
- ***Deffered Re-weighting(DRW) & Deffered Re-sampling(DRS)***<br>
*better than conventional one-stage methods*
  - 더 나은 Feature에서 시작해서 ***adjust the decision boundary and locally tunes features***
- ***Decomposing representation and classifier learning***<br>
  - **먼저 *Instance-balanced sampling*으로 *Deep Model*을 학습**
    - 그 후 *Parameters of Representation learning*이 고정된 *Class-balanced sampling*으로 *classifier*를 미세 조정
  - ***The cumulative learning strategy***<br>
    - *bridge the representation learning and classifier re-balancing*
    - *requires dual samplers of instance-balanced and reversed instance- balanced sampler*

---

### 3.1 Study of mixup Strategy

***Instance-balanced sampling & mixup***<br>
*Instance-balanced sampling : **The most general representation among all for long-tailed recognition***<br>
*mixup : **The Network trained with mixup are better calibrated***<br>

- ***Mixup in the Two-stage Decoupling framework***
  - *Higher representation generalization*
  - *reduce Over-confidence*
<br><br>

***Stage 1***<br>
180epochs 동안 *ImageNet-LT*에서 *Original Cross-entropy Model, Two stage Models of cRT and LWS* 학습시킨다 <br><br>
***Stage 2***<br>
각각 10epochs에 대해 미세 조정한다.
<br><br>
두 단계에 대한 *Training setup(with/without mixup alpha = 0.2)*을 변경한다. <br><br>*체크표시는 mixup이 적용했을 때 x표시는 mixup을 적용하지 않을 때이다.*<br>(Top-1 Accuracy / ECE에 대한 표)
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-6.png?raw=1" width = "800" ></p>

- *Mixup*을 적용했을 때<br>
  - *Improvement of Cross Entropy*는 ***can be ignored***
  - *Stage 1*에서 *cRT, LWS* 모두 **성능이 크게 향상**
  - *Stage 2*에서 추가로 *mixup*을 진행시 **개선 효과가 없거나 오히려 성능을 손상시킴**
  
**위 결과에 대한 이유를 설명하는 mixup의 역할** 
<br>1. *encourages representation learning*
<br>2. *but, adverse or negligible effect on classifier learning*<br><br>
**정리**<br>
즉, 1단계에서 *mixup*은 representation learning에는 좋은 효과를 보여주지만 <br>**2단계에서는 *classifier learning*에서는 효과를 보여주지 못하거나 오히려 악영향을 미친다.**<br>
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-7.png?raw=1" width = "800" ></p>
위 그림은 ***Final classifier weight norms***을 확인한 것이다. 위 그림을 봤을 때 *mixup*이 *tail classes*에 더 우호적일 수 있다는 것을 보여준다.<br><br>

그래서 2단계에서 *mixup*을 추가했을 때 생기는 불안정한 결과를 개선하기 위한 방법으로 다음과 같은 방법을 추가로 제안한다.<br>
***-> Label-aware smoothing***

### 3.2. Label-aware smoothing

*Cross-entropy*의 최적의 솔루션과 비교하여,<br><br>***Label-aware smoothing***<br>
- ***encourage a finite output, more general and remedying overfit***

<br><br>
또한 인스턴스 수가 더 많은 클래스인 Head class가 
<br>더 다양한 예를 많이 포함하고 있기 때문에 예측 확률이 *Tail class*보다 더 좋다.<br>
따라서 더 큰 ***Label smoothing factor***를 부여해야 한다고 논문에서는 말하고 있다.<br><br>

그리고 *Label-aware smoothing*은 *Cross-entropy*보다 더 복잡하기 때문에
<br>***Generalized classifier learning framework***에 적용해야 한다고 말하고 있고<br>
예시로는 위에서 배운 ***cRT*** 혹은 ***LWS***를 말한다.<br><br>

*cRT*와 *LWS*중에서 대규모 데이터셋에서 *LWS*가 더 좋은 결과를 주기 때문에<br>
실험에서는 ***LWS + Label-aware smoothing***으로 결과를 확인한다. <br>
결과는 아래 그림과 같다.
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-10.png?raw=1" width = "800" ></p>
왼쪽부터 *Head, Medium,Tail* 순으로 결과를 보여준 것이다.<br>
(연한 파랑 : *LWS + Cross-Entropy* , 짙은 파랑 : *LWS + Label-Aware Smoothing*)
<br><br>
***LWS + Cross-Entropy***의 경우엔 <br>*Head*와 *Medium*에서 실제로 1.0에 가까울 정도로 높은 *Over-confident*를 보이는 반면에,<br><br>
***LWS + Label-Aware Smoothing***의 경우엔 <br> *Over-confident*가 많이 감소한 것을 확인할 수 있다.<br><br>

### 3.3. Shift Learning on Batch Normalization

인스턴스 균형 샘플링으로 1단계에서 학습한 후 클래스 균형 샘플링으로 2단계에서 학습한다.<br><br>
위 *Two-stage training framework*는 ***Transfer learning*의 변형**으로 볼 수 있는데<br><br>
Transfer learning 관점에서 Two-stage training framework를 보면<br><br> backbone부분을 고정하고 Classifier를 튜닝하는 것은 unreasonable 하다. <br><br>
다른 샘플링 방법이기 때문에 *Head, Medium, Tail* 구성 비율이 다르고, 따라서 *Bias*가 존재한다.<br><br>
2가지 방법을 참고해서 사용<br>
- ***AdaBN & TransNorm***
  - update the running mean μ and variance σ
  - fix the learnable linear transformation parameters α and β for better normalization in Stage-2

---

## 4. Experiments

### 4.1.1 Datasets and Setup

***1. CIFAR-10 and CIFAR-100***<br>
50000장 Training & 10000장 Validation + 10개 카테고리 혹은 100개 카테고리<br>
***Long-tailed Dataset 사용***<br><br>

***2. ImageNet-LT and Places-LT***<br>
- ImageNet-LT<br>
115800 이미지 + 100 카테고리 (class cardinality:5~1280)
- Places-LT<br>
184500 이미지 + 365 카테고리 (class cardinality:5~4980)

***3. iNaturalist 2018***<br>
437500 이미지 + 8142 카테고리

### 4.1.2 Implementation Details

***SGD optimizer with momentum = 0.9 to optimize network***<br>
- MiSLAS model with ResNet-32 + 160~180 epochs에서 0.1로 learning rate 감소<br>
- Use cosine learning rate -> MiSLAS model + ResNet- 10, 50, 101, 152

### 4.2 Ablation Study

***Calibration performance***
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-11.png?raw=1" width = "800" ></p>
CIFAR-100-LT with IF 100 데이터셋으로 했을 때 <br>Calibration performance에 대한 결과이다.<br>
**본 논문에서 제시하는 MiSLAS 모델일 때가 가장 Confidence gap이 적은 것을 보여준다.**<br><br>
***Comparing re-weighting with label-aware smoothing***
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-12.png?raw=1" width = "800" ></p>
class balanced cross-entropy와 Label-aware smoothing을 비교했을 때 결과이다.<br>
위 결과에서 알 수 있듯이 Label-aware smoothing을 했을 때 <br>***Over-confidence*도 크게 감소**하고 ***Accuracy*도 상승**하는 것을 확인할 수 있다.

### 4.2.1 Result

<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-13.png?raw=1" width = "800" ></p>
**위 결과 표를 통해 알 수 있는 점**
- 1단계에서 mixup을 했을 때 Accuracy증가 + ECE 감소
- Shift learning on BN + Label-aware smoothing까지 했을 때 Accuracy 약간 증가 + ECE 크게 감소

<br>
***Comparison with State-of-the-arts***
<p align="center"><img src="https://github.com/Changhyun-song/Changhyun-song.github.io/blob/main/_posts/images/paper_review/paper_review1/paper_review1-14.png?raw=1" width = "800" ></p>
<br><br>
전체적으로 본 논문 이전에 사용되었던 방법들이랑 비교했을 때 <br>***MiSLAS가 압도적으로 높은 Accuracy + 좋은 Calibration임을 보여준다.***
<br>
대규모 데이터 셋인 a,b,c에서도 MiSLAS가 높은 성능을 가지고 있는 것을 확인할 수 있다.

---

### 5 Conclusion

1. ***Long-tailed Dataset을 학습한 모델은*** <br>balanced dataset을 학습한 모델보다 ***miscalibrated and overconfident***
2. 첫 번째 솔루션 - ***Mixup***
- 1단계에서 mixup 사용 -> ***representation learning에서 좋은 효과***(classifier learning에서는 오히려 역효과)
3. 두 번째 솔루션 - ***Label-aware smoothing***
- LWS를 사용하여 Over-confidence를 크게 감소시킨다.
4. 세 번째 솔루션 - ***Shift learning on the batch normaization***
- Two-stage method framework에서 ***Dataset bias를 줄이기 위해서*** 사용 -> 성능 향상
5. 논문에서 제시하는 ***MiSLAS 모델이 가장 좋은 Accuracy + Calibration*** 을 보여줌
- 대규모 데이터 셋에서도 마찬가지 좋은 성능을 보여줌