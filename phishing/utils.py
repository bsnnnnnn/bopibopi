from transformers import AutoModel, AutoTokenizer
import torch
import random

def load_kcbert_model():
    # KcBERT tokenizer가 있는 디렉토리 경로 지정
    tokenizer_directory = "C:\\Users\\03123\\.cache\\huggingface\\hub\\models--beomi--kcbert-base\\snapshots\\0f2f3f8ce58a3e2dab3f4c9f547cbb612061c2ed"  # 슬래시 방향 수정
    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)
    model = AutoModel.from_pretrained(tokenizer_directory)
    return model, tokenizer

def calculate_embedding(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt",max_length=128, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    embedding = embedding.astype(float)
    return embedding

############## mel_spectrogram
import librosa
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import os
from PIL import Image
import cv2

# 오디오 파일 멜 스펙트로그램, MFCC로 변환
def convert_audio_to_mel_spectrogram(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # dB 스케일로 변환

    # 현재 시간 기반으로 한 이름 생성
    timestamp = int(time.time())
    mel_spectrogram_img_path = f'media\mel_spectrograms_img\mel_spectrogram_img_{timestamp}.png'
    mel_spectrogram_np_path = f'media\mel_spectrograms_np\mel_spectrogram_np_{timestamp}.npy'
    mfcc_path = f'media\mfcc\mfcc_{timestamp}.npy'

    # 넘파이로 저장
    np.save(mel_spectrogram_np_path, mel_spectrogram_db)
    np.save(mfcc_path, mfcc)

    # 이미지로 저장
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000) #fmax : 주파수의 최대값
    plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram')
    plt.savefig(mel_spectrogram_img_path)

    return mel_spectrogram_img_path, mel_spectrogram_np_path, mfcc_path

# resnet 모델 
def load_resnet_model():
    resnet_path = "phishing\model"
    model = tf.keras.models.load_model(resnet_path)
    return model

# 전화번호 랜덤 생성
def get_rand_numbers():
    numbers = '0123456789'
    num1 = "".join(random.sample(numbers, 4))
    num2 = "".join(random.sample(numbers, 4))

    phone_num = f"010-{num1}-{num2}"

    return phone_num