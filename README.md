# Speech Recognition — DeepSpeech2 CNN + Bidirectional GRU + CTC

End-to-end automatic speech recognition system built from 
scratch in PyTorch, converting raw audio to transcribed 
text using Residual CNNs, Bidirectional GRUs, and CTC loss.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Audio](https://img.shields.io/badge/Library-torchaudio-green)
![Task](https://img.shields.io/badge/Task-Speech%20Recognition-purple)

---

## Overview

Implements a DeepSpeech2-inspired ASR pipeline: raw audio 
is converted to 128-band Mel Spectrograms with SpecAugment 
augmentation, processed through 3 Residual CNN blocks and 
5 Bidirectional GRU layers, and decoded using CTC loss — 
eliminating the need for manually aligned audio-to-character 
labels. Trained and validated on the LibriSpeech dataset.

---

## Dataset

- **Name:** LibriSpeech ASR Corpus
- **Source:** Downloaded automatically via torchaudio —
  no manual download needed
- **Training split:** `dev-clean`
- **Validation split:** `test-clean`
- **Characters:** 28 classes — 26 letters + apostrophe 
  + space + 1 CTC blank token (index 28)

---

## Model Architecture

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Scheduler | OneCycleLR (linear annealing) |
| Loss | CTCLoss (blank=28) |
| Learning Rate | 1e-5 |
| Batch Size | 8 |
| Epochs | 1 (increase for better results) |
| Device | CUDA / CPU (auto-detected) |

---


## Technologies Used

Python, PyTorch, torchaudio, NumPy, Matplotlib, tqdm

---

## How to Run

```bash
git clone https://github.com/OyelolaIbrahim/speech-recognition-deepspeech2.git
cd speech-recognition-deepspeech2
pip install -r requirements.txt
jupyter notebook speech_to_text.ipynb
```

> **Note:** GPU strongly recommended. LibriSpeech 
> (~700 MB) downloads automatically on first run.

