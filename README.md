# 독초 및 비독초 판별 프로젝트

## 목차
1. [서론](#서론)
2. [이론적 배경](#이론적-배경)
3. [데이터셋 설명](#데이터셋-설명)
4. [모델 설계 및 구현](#모델-설계-및-구현)
5. [모델 성능 평가](#모델-성능-평가)
6. [결과 분석 및 논의](#결과-분석-및-논의)
7. [참고문헌](#참고문헌)
8. [부록](#부록)

## 서론
### 1. 문제 정의 및 연구 배경
#### 가) 연구 배경
최근 몇 년 동안 자연에서 얻을 수 있는 다양한 식물들에 대한 관심이 높아지면서 약초와 식용 식물을 찾는 사람들이 늘어나고 있습니다. 그러나 일부 식물은 독성을 지니고 있어 잘못 섭취할 경우 심각한 건강 문제를 일으킬 수 있습니다. 예를 들어, 2021년 4월 강원도에서 독초를 산삼으로 오인하여 섭취한 노인이 중독 증상을 호소한 사건이 보도되었습니다. 이러한 독초 섭취로 인한 사고는 예기치 않게 발생할 수 있으며, 특히 노인층이나 식물에 대한 지식이 부족한 사람들에게 큰 위험을 초래할 수 있습니다.![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/951a0cf3-0c80-4d5c-aed7-7a9fd6a4573c)


강원도 소방본부에 따르면, 강원 지역에서 봄철 독초 중독 사고는 2021년 이후 매년 20건 이상 꾸준히 발생하고 있습니다. 또한, 식품의약품안전처의 소비자 위해감지시스템(CISS) 데이터에 따르면, 2020년부터 2022년까지 총 26건의 독초 섭취 사고가 보고되었습니다. 이러한 사고는 독초와 무독성 식물이 외관상 매우 유사하여 일반인이 이를 구별하기 어렵기 때문에 발생합니다.

독초 섭취 사고를 줄이기 위해서는 독초와 무독성 식물을 정확하게 구별할 수 있는 기술적 방법이 필요합니다. 전통적으로는 식물학적 지식과 경험에 의존하여 독초를 식별했으나, 이는 시간이 많이 걸리고 전문가의 도움 없이 일반인이 활용하기에는 한계가 있습니다. 따라서, 현대 기술을 활용하여 독초를 정확하게 식별할 수 있는 시스템을 개발하는 것이 필요합니다.

#### 나) 문제 정의
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/3405e6cf-580f-4749-ab9b-b7cb22b2b072)

본 연구는 인공지능(AI) 기술을 활용하여 독초와 무독성 식물을 구별하는 모델을 개발하는 것을 목표로 했습니다. 특히, Convolutional Neural Networks(CNN)와 ResNet50 모델을 사용하여 이미지를 기반으로 독초를 분류하는 시스템을 구축하고자 합니다. CNN과 ResNet50은 이미지 인식 분야에서 높은 성능을 보이는 딥러닝 모델로, 복잡한 패턴 인식과 분류에 탁월한 능력을 가지고 있습니다.

연구 목표는 다음과 같습니다.
1. 독초와 무독성 식물의 정확한 분류
2. CNN과 ResNet50 모델의 성능 비교 및 향상
3. 실시간 사용 가능한 시스템 개발

#### 다) 연구 목적
딥러닝 기술을 활용하여 독초와 비독초를 분류하는 인공지능 모델을 개발하는 연구 목적은 다음과 같이 구체화할 수 있습니다.
1. 독초와 비독초를 정확하게 분류할 수 있는 모델 개발
2. 다양한 딥러닝 모델을 비교하여 최적의 모델 선정
3. 데이터셋 구축 및 모델 성능 평가
4. 실제 응용 가능성 탐색
5. 교육 및 학습 자료로 활용
6. 연구의 한계점 인식 및 개선 방안 제시

## 이론적 배경
### 1. CNN(Convolutional Neural Network)의 원리
Convolutional Neural Network(CNN)은 이미지 인식, 분류, 객체 탐지 등에 널리 사용되는 인공 신경망의 한 종류입니다. CNN의 핵심 개념은 '합성곱(convolution)'으로, 이미지의 특정 특징을 추출하는 역할을 합니다. 주요 구성 요소는 다음과 같습니다.

1. 합성곱층(Convolutional Layer)
2. 활성화 함수(Activation Function)
3. 폴링층(Pooling Layer)
4. 완전 연결층(Fully Connected Layer)

### 2. ResNet50(Residual Network)의 원리
ResNet(Residual Network)은 깊은 신경망을 효과적으로 학습할 수 있는 방법을 제시한 혁신적인 네트워크입니다. ResNet의 핵심 아이디어는 "잔차 학습(residual learning)"으로, 이는 네트워크가 직접 원하는 매핑을 학습하는 대신, 잔차를 학습하도록 하는 것입니다. ResNet50의 주요 구성 요소는 다음과 같습니다.

1. 합성곱층(Convolutional Layer)
2. Batch Normalization
3. 활성화 함수(ReLU)
4. 스킵 연결(Skip Connection)

### 3. 독초 및 비독초 판별의 필요성
독초와 비독초를 정확히 구분하는 능력은 공공 안전과 보건 측면에서 매우 중요합니다. 잘못된 식물 섭취는 심각한 건강 문제를 일으킬 수 있으며, 최악의 경우 생명까지 위협할 수 있습니다. 독초와 비독초를 구분하는 연구는 계속되어야 하며, 이를 통해 독초 섭취로 인한 사고를 예방하고, 공공의 건강을 보호하며, 대중의 인식을 높이는 데 기여할 수 있습니다.

## 데이터셋 설명
### 1. 데이터셋 출처 및 구성
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/1c182d5b-de23-4c0b-b2c3-fa37448e807b)

본 연구에 사용된 데이터셋은 AI Hub에서 제공하는 동의보감 독초 판별 이미지 데이터셋입니다. 이 데이터셋은 동의보감에 기재된 다양한 독초와 그 유사 식물들의 이미지를 포함하고 있으며, 각 이미지에 대해 독성 여부를 판단할 수 있는 정보를 제공합니다. 주요 폴더는 다음과 같습니다.

1. 원천 데이터
2. 라벨링 데이터
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/df354596-1e8e-43f6-b230-a62d758aefc8)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/c5d40842-2e6f-41ef-8b4c-a41d8a0b173c)


### 2. 데이터 전처리 과정
데이터 전처리는 데이터셋을 머신러닝 모델에 적합하게 변환하는 중요한 과정입니다. 원본 데이터셋은 다양한 형태와 품질의 이미지를 포함하고 있어, 이를 그대로 모델에 입력할 경우 학습 성능이 저하될 수 있습니다. 전처리 과정은 다음과 같습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/136c5e45-2357-4f89-96bc-07a8978d96a0)

1. 데이터 정제
2. 데이터 변환
3. 라벨링 정보 처리
4. 데이터셋 분할

### 3. 라벨링 기준 및 방법
라벨링은 데이터셋의 각 이미지에 대해 정답 값을 부여하는 과정으로, 머신러닝 모델 학습에 있어서 매우 중요한 단계입니다. 본 연구에서는 다음과 같은 기준을 적용하였습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/d98e6d7f-fbf8-405b-b3b9-01d80639f2da)

1. 독성 여부
2. 유사 식물 구분
3. 식물 부위

## 모델 설계 및 구현
### 1. CNN 모델 설계
Convolutional Neural Network(CNN)는 이미지 처리 및 분석에 널리 사용되는 딥러닝 모델입니다. 본 연구에서는 독초와 비독초를 분류하기 위해 CNN을 설계하였습니다. 주요 구성 요소는 다음과 같습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/7fd351ad-afd8-4b9a-b8a7-a80bcb27b518)

1. 합성곱층
2. 풀링층
3. 활성화 함수
4. 배치 정규화
5. 드롭아웃
6. 완전 연결층

### 2. ResNet50 모델 설계
ResNet(Residual Network)은 깊은 네트워크 구조를 가지고 있어 다양한 특성을 학습할 수 있는 능력을 가지고 있습니다. ResNet50은 이러한 ResNet 구조 중 하나로, 50개의 층으로 구성되어 있습니다. 본 연구에서는 ResNet50을 사용하여 독초와 비독초를 분류하는 모델을 설계하였습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/257e8442-4282-4580-a53d-20fa1602f627)

### 3. 모델 학습 환경 및 도구
본 연구에서는 다음과 같은 학습 환경을 구축하여 모델 학습을 진행하였습니다.

1. 하드웨어 환경
2. 소프트웨어 도구
3. 모델 학습 과정

### 4. 하이퍼파라미터 설정
적절한 하이퍼파라미터 설정은 모델의 최적 성능을 달성하기 위해 필수적입니다. 주요 하이퍼파라미터에는 학습률(Learning Rate), 배치 크기(Batch Size), 에포크 수(Number of Epochs), 옵티마이저(Optimizer) 등이 있습니다.

## 모델 성능 평가
### 1. 성능 평가 지표 설명
다양한 성능 평가 지표가 존재하며, 각 지표는 특정 상황에서 모델의 성능을 평가하는 데 유용합니다. 본 연구에서는 주요 성능 평가 지표인 정확도(Accuracy), F1 스코어(F1 Score), 정밀도(Precision), 재현율(Recall)을 사용하였습니다.![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/fa157c8c-f567-44f3-96fa-6e92164eb31e)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/0c3b49d4-1584-4b23-a290-0bc958edfbeb)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/c254b925-8e87-42fc-bcfe-311ba3194eb7)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/640e7a1e-7f4a-470c-9c9f-298c5aa7e63d)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/7f4e8874-3a35-487c-81cb-4655d454c7f0)


### 2. 모델 학습 및 평가 결과
본 연구에서는 CNN과 ResNet50 두 가지 모델을 사용하여 독초 분류 문제를 해결하고자 하였습니다. 각 모델은 독초와 비독초를 정확히 분류하기 위해 학습되었으며, 성능 평가 지표를 사용하여 각 모델의 성능을 평가하였습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/3c309755-2bf6-4396-9484-f1fcf96f2a04)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/dbdb3417-099f-4c9f-9975-f5e452771788)

### 3. 성능 비교: CNN vs ResNet50
CNN 모델과 ResNet50 모델의 성능을 비교한 결과, ResNet50 모델이 모든 성능 지표에서 CNN 모델보다 우수한 성능을 보였습니다. ResNet50 모델은 더 깊은 네트워크 구조와 사전 학습된 가중치를 통해 다양한 특성을 학습하고, 일반화 능력을 극대화할 수 있음을 확인할 수 있습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/cd9d4b64-50f6-4240-8b40-89f050308d91)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/5bb3ccb2-63e8-4486-8721-0b08a67218a9)

## 결과 분석 및 논의
### 1. 각 모델의 성능 분석
CNN 모델과 ResNet50 모델의 성능을 다양한 성능 지표를 통해 분석하고 비교하였습니다. ResNet50 모델은 더 높은 정확도와 F1 스코어를 기록하며, CNN 모델보다 우수한 성능을 보였습니다.
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/2070afae-b2d0-479a-bb1c-9144fe939e51)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/c21354d3-d73d-473d-9e5e-d82fc32d2647)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/d7a5c888-d50f-4d2a-8e89-18ad0307a1b9)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/ccab86e4-e2a3-450a-ba81-0dffc8a65447)
![image](https://github.com/tcpu1005/ResnetPoisonousHerbIdentifier/assets/106582928/b79382e6-735b-494e-af60-a78d22300b9e)

### 2. 결과에 대한 논의
CNN 모델은 과적합 문제와 모델의 복잡성 부족으로 인해 제한적인 성능을 보였습니다. 반면, ResNet50 모델은 높은 일반화 능력과 복잡한 특성 학습 능력으로 뛰어난 성능을 보였습니다.

### 3. 모델의 한계점 및 개선 방안
모델의 성능을 평가한 결과, ResNet50 모델이 CNN 모델보다 우수한 성능을 보였습니다. 그러나 두 모델 모두 몇 가지 한계점을 가지고 있으며, 이를 개선하기 위한 방안을 모색할 필요가 있습니다.

## 참고문헌
1. AI 허브 데이터셋
   - AI 허브. "동의보감 약초 이미지 AI 데이터". AI 허브.
2. GitHub 프로젝트
   - Ban-Ursus. "Discriminator for distinguishing between poisonous and medicinal herbs". GitHub.
3. 논문: Deep Residual Learning for Image Recognition
   - He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. Microsoft Research.
4. 논문: ImageNet Classification with Deep Convolutional Neural Networks
   - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. University of Toronto.
5. 논문: Deep residual learning for image recognition
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.

## 부록
### 1. 코드 및 추가 자료
#### a) Layer Added CNN 코드
```python
# 필요한 모듈 임포트
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# NVIDIA GPU 종류 확인
# !nvidia-smi

class CustomDataset(Dataset):
    def __init__(self, src_base_folder, label_base_folder, transform=None):
        self.src_base_folder = src_base_folder
        self.label_base_folder = label_base_folder
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # src_base_folder 내의 모든 식물 폴더를 순회
        for plant_folder in os.listdir(src_base_folder):  # plant_folder : 꽃 열매 잎 전초
            if plant_folder.startswith("VS_"):
                plant_type = plant_folder[3:]  # 식물 이름 ex) VS_가시박 VS_가지... "VS_" 뒤부터 식물 이름.
                label_plant_folder = f"VL_{plant_type}"  # 라벨 폴더 이름 ex) VL_가시박 VL_가지 ... "VL_" 뒤부터 식물 이름.
                parts_folders = os.listdir(os.path.join(src_base_folder, plant_folder))  # parts_folders : VS_가시박/꽃 VS_가시박/열매...

                # 각 부위 폴더(꽃, 열매, 잎, 전초) 내부를 순회
                for part_folder in parts_folders:
                    src_folder = os.path.join(src_base_folder, plant_folder, part_folder)  # 원천 데이터/VS_가시박/꽃 , 원천 데이터/VS_가시박/열매 ...
                    label_folder = os.path.join(label_base_folder, label_plant_folder, part_folder)  # 라벨링 데이터/VL_가시박/꽃 , 라벨링 데이터/VL_가시박/열매 ...

                    # 이미지와 해당 JSON 파일을 매칭
                    for image_name in os.listdir(src_folder):  # 원천 데이터/VS_가시박/꽃 폴더 내 순회
                        image_path = os.path.join(src_folder, image_name)  # 조건문 밖에서 image_path 변수 정의
                        if image_name.endswith('.jpg'):  # .jpg로 끝나는 파일 즉 이미지 파일이면
                            label_name = image_name.replace('.jpg', '.json')  # ex) 가시박_꽃_1528579.json으로 label_name (jpg 만 json으로 교체)
                            label_path = os.path.join(label_folder, label_name)  # ex) 원천 데이터/VS_가시박/꽃/가시박_꽃_1528579.json

                            if os.path.exists(label_path):
                                with open(label_path, 'r', encoding='utf-8') as file:  # 인코딩 추가
                                    data = json.load(file)
                                    toxic_info = data['plantinfo']['toxic_info']
                                    if toxic_info == 'Y':
                                        label = 1  # 독초
                                    else:
                                        label = 0  # 비독초

                                    self.image_paths.append(image_path)
                                    self.labels.append(label)
                            else:
                                print(f'라벨 파일이 없습니다: {label_path}')
                        else:
                            print(f'이미지 파일이 없습니다: {image_path}')

        if len(self.image_paths) == 0 or len(self.labels) == 0:
            raise ValueError("데이터셋에 이미지나 라벨이 없습니다.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # model의 input이 되는 images ex) 3*6705*4475 resolution은 image별로 상이함.

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 이미지 전처리 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)), # image resolution 224*224로 모두 동일화.
    transforms.ToTensor(), #image tesnor화 및 정규화
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 데이터셋 경로 설정
src_base_path = "180.동의보감 약초 이미지 AI데이터/01.데이터/2.Validation/원천데이터"
label_base_path = "180.동의보감 약초 이미지 AI데이터/01.데이터/2.Validation/라벨링데이터"

# 데이터셋 로딩
full_dataset = CustomDataset(src_base_path, label_base_path, transform=transform)

# 데이터셋 분할: 훈련 60%, 검증 20%, 테스트 20%
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
####### seed값 지정해서 같은 데이터셋에 대해서 모델 성능평가###
generator2 = torch.Generator().manual_seed(42)
############################################################## generator2 상관없음 걍 변수명
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size],generator=generator2)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class Block(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim1, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class DeeperCNN(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.block1 = Block(3, 32, 64, 128)
        self.block2 = Block(128, 64, 128, 256)
        self.block3 = Block(256, 128, 256, 512)
        self.block4 = Block(512, 256, 512, 512)
        # self.block5 = Block(512, 256, 128, 64)
        # self.block6 = Block(64, 128, 64, 32)
        # self.block7 = Block(32, 64, 32, 16)
        # self.block8 = Block(16, 32, 16, 8)
        # self.block9 = Block(8, 16, 8, 4)
        # self.block10 = Block(4, 8, 4, 2)
        
        self.fc1 = nn.Linear(512*14*14, 1024) #224 → 112 → 56 → 28 → 14 → 7 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        # x = self.block9(x)
        # x = self.block10(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# class DeeperCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super(DeeperCNN, self).__init__()

#         self.layers = nn.ModuleList()
#         in_channels = 3
#         out_channels = 64

#         # 합성곱 층을 30개로 만들기 위해 반복문을 사용하여 레이어 추가
#         for i in range(30):
#             self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             self.layers.append(nn.BatchNorm2d(out_channels))
#             self.layers.append(nn.ReLU(inplace=True))
#             in_channels = out_channels
#             if (i + 1) % 10 == 0 and out_channels < 512:
#                 out_channels *= 2

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(out_channels, num_classes)

#     def forward(self, x):
#         for i in range(30):
#             x = self.layers[3 * i](x)
#             x = self.layers[3 * i + 1](x)
#             x = self.layers[3 * i + 2](x)
#             if (i + 1) % 10 == 0:
#                 x = F.max_pool2d(x, kernel_size=2, stride=2)
        
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

model = DeeperCNN().to(device)
# 손실 함수와 최적화 함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 모델 훈련 함수
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    #변수 정의 -수정
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    #수정
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_preds = [] #추가
        all_labels = [] #추가

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #추가
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #추가
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        #f1 추가
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        train_f1_scores.append(epoch_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        #추가
        all_preds = []
        all_labels = []
        

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                #추가
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_dataset)
            val_acc = val_corrects.double() / len(val_dataset)
            #f1 추가
            val_f1 = f1_score(all_labels, all_preds, average='weighted')

            # 각 에포크의 손실, 정확도, F1 점수를 리스트에 추가 (값 추가)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc.item())
            val_f1_scores.append(val_f1)
                
            print(f'Epoch {epoch+1}/{num_epochs} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = model.state_dict()
                torch.save(model.state_dict(), '김지훈/best_model_cnn.pth')

    print(f'Best Val Acc: {best_acc:.4f}')

    # 그래프 그리기
    epochs = range(1, num_epochs + 10)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1_scores, label='Train F1 Score')
    plt.plot(epochs, val_f1_scores, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return model

# 테스트 세트에서 모델 성능 평가 및 F1 점수 계산
def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = running_corrects.double() / total
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'F1 Score: {f1:.4f}')

# 모델 훈련
trained_model = train_model(model, criterion, optimizer, num_epochs=10) 
evaluate_model(trained_model, test_loader)

# 예측 함수 정의
def predict_toxicity(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0][preds[0]]

    class_names = ['비독초', '독초']
    return class_names[preds[0]], probability.item()

# 예측을 위해 샘플 이미지 사용
sample_image_path = "180.동의보감 약초 이미지 AI데이터/01.데이터/2.Validation/원천데이터/VS_가시박/꽃/가시박_꽃_1528579.jpg"  # 샘플 이미지 경로
predicted_class, probability = predict_toxicity(trained_model, sample_image_path, transform)
print(f'Predicted Class: {predicted_class}, Probability: {probability:.4f}')
