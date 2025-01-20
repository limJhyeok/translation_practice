# Translation Practice

**언어 선택 / Language Selection:**

- [🇰🇷 한국어 (Korean)](README.md)
- [🇺🇸 English](README.en.md)

This repository is designed to study machine translation using Seq2Seq and Attention mechanisms commonly used in natural language processing. It covers data processing, model construction, training, and evaluation using PyTorch.

## Dataset Used
- English-French translation pairs are used, utilizing files provided by [Tatoeba](https://tatoeba.org/eng/downloads) and [ManyThings](https://www.manythings.org/anki/).

  The dataset files are already included in the **data** folder, so no additional downloads are required.
  
- **How to Download the Data (Optional)**:
    
    If you need to download the data again, you can use the following commands to download and extract the files:
    ```bash
    wget https://download.pytorch.org/tutorial/data.zip
    unzip data.zip -d data/
    ```
    - The translation pair file (`data/eng-fra.txt`) is tab-separated:
      ```
      I am cold.   J'ai froid.
      ```

## 📁 Project Structure

### 1. **[Seq2Seq_practice.ipynb](https://github.com/limJhyeok/translation_practice/blob/main/Seq2Seq_practice.ipynb)**
- **Key Contents**:
  - Implements a basic Seq2Seq model to train on a small dataset (a single pair of sentences).
- **Purpose**:
  - To verify that the model works correctly in a simple sanity check stage.

### 2. **[Seq2Seq.ipynb](https://github.com/limJhyeok/translation_practice/blob/main/Seq2Seq.ipynb)**
- **Key Contents**:
  - Implements a complete Seq2Seq model designed to handle larger datasets.
- **Key Features**:
    - Data preprocessing
    - Embedding, tokenization, RNN
    - Seq2Seq model architecture (Encoder, Decoder, Context Vector)
    - Train & evaluation loop using PyTorch

### 3. **[AttnSeq2Seq.ipynb](https://github.com/limJhyeok/translation_practice/blob/main/AttnSeq2Seq.ipynb)**
- **Key Contents**:
  - Enhances the Seq2Seq model by introducing the Attention mechanism.
- **Key Features**:
    - Background and motivation for Attention Seq2Seq
    - Improved context representation using BiRNN (Bidirectional RNN)
    - Explanation of the Attention mechanism
    - Train & evaluation loop using PyTorch

## 🔧 How to Run
### 1. Download the Repository and Move to Google Drive

1. Download the repository:
    
    ```bash
    git clone https://github.com/username/translation-practice.git
    ```
    
2. Rename the downloaded folder to `translation` and upload it to Google Drive.
    - Example: `/content/drive/My Drive/translation`

### 2. Open in Google Colab

1. Mount Google Drive to Colab:
    
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    FOLDERNAME = 'translation'
    sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))
    %cd /content/drive/My Drive/$FOLDERNAME
    ```
    
2. Open the `.ipynb` file by navigating to its path:
    - Example: `/content/drive/My Drive/translation/Seq2Seq.ipynb`

## 📖 Learning Goals
1. Understand Seq2Seq and Attention-based translation model architectures
2. Implement data preprocessing, model training, and inference using PyTorch
3. Study the effects of BiRNN and the Attention mechanism

## Key References
Please refer to the references provided in the notebooks. 
