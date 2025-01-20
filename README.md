# Translation Practice

이 리포지토리는 자연어 처리에서 사용되는 Seq2Seq 및 Attention 모델을 활용하여 번역 작업을 학습하기 위해 작성되었습니다. PyTorch를 사용하여 데이터 처리, 모델 구성, 학습 및 평가 과정을 다룹니다.

## 사용한 데이터
  - 영어-프랑스어 번역 쌍 데이터를 사용하며, [Tatoeba](https://tatoeba.org/eng/downloads)와 [ManyThings](https://www.manythings.org/anki/)에서 제공된 파일을 활용합니다.
  - **데이터 다운로드**:  
    ```
    wget https://download.pytorch.org/tutorial/data.zip
    unzip data.zip -d data/
    ```
  - 번역 쌍 파일(`data/eng-fra.txt`)은 탭으로 구분된 형식입니다.
    ```
    I am cold.   J'ai froid.
    ```
## 📁 프로젝트 구성

### 1. **Seq2Seq_practice.ipynb**
- **주요 내용**
  - Seq2Seq 기본 구조를 구현하여 대량의 데이터를 학습시키기 전 한 쌍의 번역 문장을 학습해보는 노트북입니다.
- **구현 이유**
  - 올바르게 학습이 되는지 간단히 확인(sanity check)하기 위해

### 1. **Seq2Seq.ipynb**
- **주요 내용**
  - 데이터 다운로드 및 전처리 과정
  - Seq2Seq 모델 구조 설명 (Encoder, Decoder, Context Vector)
  - 임베딩, 토크나이즈, RNN
  - PyTorch를 활용한 학습 및 평가 방법
  
---

### 3. **AttnSeq2Seq.ipynb**
- **주요 내용**
  - **Attention Seq2Seq**의 탄생 배경과 동작 원리를 설명합니다.
  - BiRNN(양방향 RNN) 인코더를 활용해 컨텍스트를 풍부하게 처리
  - Attention 메커니즘을 통해 디코딩 시점마다 가중치를 동적으로 계산
  - 학습 및 평가 과정 포함

---

## 🔧 실행 방법
### 1. 리포지토리 다운로드 및 Google Drive로 이동

1. 리포지토리를 다운로드합니다:
    
    ```bash
    git clone https://github.com/username/translation-practice.git
    
    ```
    
2. 다운로드한 폴더를 `translation`으로 이름을 바꾸어서 Google Drive에 업로드합니다.
    - 예: `/content/drive/My Drive/translation`

### 2. Google Colab에서 열기

1. Google Drive를 Colab과 연동:
    
    ```python
    drive.mount('/content/drive')

    FOLDERNAME = 'translation'
    sys.path.append('content/drive/My Drive/{}'.format(FOLDERNAME))
    %cd /content/drive/My Drive/$FOLDERNAME
    ```
    
2. `.ipynb` 파일 경로를 확인하고 열기:
    - 예: `/content/drive/My Drive/translation-practice/Seq2Seq.ipynb`

## 📖 학습 목표
1. Seq2Seq 및 Attention 기반 번역 모델 구조 이해
2. PyTorch를 활용한 데이터 전처리, 모델 학습 및 추론 구현
3. BiRNN과 Attention 메커니즘의 효과 학습

## 주요 참고 문헌
노트북에 적혀있는 문헌 정보들을 참고해주세요
