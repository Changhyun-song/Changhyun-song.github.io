---
title: (Python 프로젝트) Finding and Measuring Lungs in CT Data (CT 데이터에서 폐 부분만 뽑아내기)
layout: post
date: '2020-09-02 03:00:00'
author: 창혀니
tags: segmentation python tensorflow keras
cover: "/assets/backgroundpic.png"
categories: python-project
---

## 1. 데이터 전처리

이번에는 CT Data에서 필요한 이미지인 폐 영역만 골라내는 segmentation을 해보도록 하겠습니다.
먼저 데이터 셋은 kaggle에서 가져오시면 됩니다.
[Dataset](https://www.kaggle.com/kmader/finding-lungs-in-ct-data?select=2d_masks.zip "데이터 셋 받는 주소")

여기서 2d_images.zip 파일과 2d_masks.zip 파일을 받아오시면 됩니다!
다운로드 받은 파일은 다음과 같이 원래 CT 이미지들이 모여있는 image 폴더와 폐 부분만 표시되어 있는 mask된 사진만 모여있는 폴더입니다.

![사진](/assets/ct-1.png)

**2d_images Dataset**

![사진](/assets/ct-2.png)

**2d_masks Dataset**

보시면 이미지파일이 tif 형식으로 되어 있습니다. Data 분야를 공부하면 할수록 느끼는 것이지만 처음 Data를 전처리하는 과정이 전체 과정의 절반이상을 차지한다고 말하는 것처럼 전처리 과정은 정확한 결과 값을 알기위해선 매우중요합니다!
따라서 우리는 이 tif 이미지들을 먼저 학습하기 쉬운 .npy 형식의 파일로 만들것입니다.

### 1-1.필요한 패키지와 함수 변수 import 하기
Jupyter notebook을 키기 전에 필요한 패기지들을 다운로드 받아줍니다. 그 후 아래코드를 넣어주시면 됩니다.
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize

import os, glob
```

### 1-2.list 형식으로 img_list와 mask_list 생성
**glob**
glob.glob 를 사용하여 우선 폴더에 있는 모든 파일명을 /*를 사용하여 모두 리스트 형식으로 불러옵니다.
```
img_list = sorted(glob.glob('2d_images/*.tif'))
mask_list = sorted(glob.glob('2d_masks/*.tif'))

print(len(img_list), len(mask_list))
```
![사진](/assets/ct-3.png)
이렇게 img_list와 mask_list에 list 형식으로 폴더에 있는 모든 파일명들을 불러온 뒤 저장합니다.

### 1-3. Array 형태로 만들고 array 형태의 이미지 확인하기
```
IMG_SIZE = 256

x_data, y_data = np.empty((2, len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

for i, img_path in enumerate(img_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    x_data[i] = img
    
for i, img_path in enumerate(mask_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    y_data[i] = img
    
y_data /= 255.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')
```
우선 imread 함수를 통해서 img_list에 있는 파일을 하나씩 불러오고 Array형태로 불러오고 img에 하나씩 저장합니다. 그 후 resize를 통해 (256,256,1) 형태로 만들어 줍니다.

![사진](/assets/ct-4.png)

x_data와 y_data를 중간에 확인해보면 총 267개의 이미지 형태인 (256,256,1)이 되는 것을 확인할 수 있다. 그리고 마지막에 squeeze 함수를 통해 1차원 배열로 바꿔 imshow 함수로 이미지를 확인한다.
이미지 형태로 잘 나오는 것을 확인할 수 있다.
![사진](/assets/ct-5.png)

마지막으로 sklearn 패키지에서 train_test_split 함수를 통해 데이터셋에서 x_train, x_val, y_train, y_val 셋을 만들어준뒤에 저장한다.
shape를 통해 마지막으로 확인해보면 train 셋 240개, val 셋 27개씩 만들어진것을 확인할 수 있다.
![사진](/assets/ct-6.png)







