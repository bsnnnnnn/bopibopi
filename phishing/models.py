from django.db import models
import pinecone
# from django.contrib.postgres.fields import ArrayField

# Create your models here.

# 유관기관 정보
class Organizations(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=20)
    image = models.CharField(max_length=255)
    call = models.CharField(max_length=50)
    work = models.CharField(max_length=100)
    url = models.CharField(max_length=255)
    category = models.CharField(max_length=20)

######딥보이스 탐지
class CallHistory(models.Model):
    id = models.AutoField(primary_key=True)
    called_at = models.DateTimeField(auto_now_add=True)
    numbers = models.CharField(max_length=50) #전화번호
    contents = models.TextField(null=True) #통화 내용
    audio_file = models.FileField(upload_to='audio/', null=True)
    results = models.BooleanField(null=True) #딥보이스 의심 여부
    # category = models.BooleanField(null=True) #기존 데이터 분류 카테고리

    # def __str__(self): #객체를 문자열로 변환할 때 어떤 문자열을 반환할지
    #     return self.title

# 멜 스펙트로그램
class MelSpectrograms(models.Model):
    id = models.AutoField(primary_key=True)
    # callhistory = models.ForeignKey(CallHistory, on_delete = models.CASCADE)
    # MFCC = models.ForeignKey(MFCC, on_delete=models.SET_NULL, null=True)
    mel_img = models.ImageField(upload_to='mel_spectrograms_img/', null=True)
    mel_np = models.FileField(upload_to='mel_spectrogram_np/', null=True)
    category = models.CharField(max_length=10, null=True)
    updated_at = models.DateTimeField(auto_now_add=True, null=True)

    def __str__(self): #객체를 문자열로 변환할 때 어떤 문자열을 반환할지
        return self.title

# mfcc
class MFCC(models.Model):
    id = models.AutoField(primary_key=True)
    # callhistory = models.ForeignKey(CallHistory, on_delete= models.CASCADE)
    # mel_spectrograms = models.ForeignKey(mel_spectrograms, on_delete=models.SET_NULL, null=True)
    mfcc = models.FileField(upload_to='mfcc/', null=True)
    category = models.CharField(max_length=10, null=True)
    updated_at = models.DateTimeField(auto_now_add=True, null=True)

# postgreSQL
# class AudioData(models.Model):
#     id = models.AutoField(primary_key=True)
#     mel_np = ArrayField(models.FloatField())
#     mfcc = ArrayField(models.FloatField())

#     class Meta:
#         db_table = 'audiodata'
#         app_label = 'phishing'
#         # using = 'second_db'

# pinecone
# class MFCC_vec(models.Model):
#     # Pinecone 벡터 필드 추가
#     id = models.AutoField(primary_key=True)
#     vector = models.JSONField(null=True, blank=True)
#     updated_at = models.DateTimeField(auto_now_add=True)

#     def save_to_pinecone(self):
#         # Pinecone에 벡터 색인
#         pinecone.create_index(index_name="mfcc", dimension=331)
#         pinecone.upsert(index_name="mfcc", ids=[str(self.id)], vectors=[self.vector])

#     # def search_similar(self):
#     #     # Pinecone를 사용하여 유사한 벡터 검색
#     #     results = pinecone.query(index_name="mel_np", query_vector=self.vector, top_k=5)
#     #     # 결과 처리...

# class mel_np(models.Model):
#     # Pinecone 벡터 필드 추가
#     id = models.AutoField(primary_key=True)
#     vector = models.JSONField(null=True, blank=True)
#     updated_at = models.DateTimeField(auto_now_add=True)
    
#     def save_to_pinecone(self):
#         # Pinecone에 벡터 색인
#         pinecone.create_index(index_name="mel_np", dimension=331)
#         pinecone.upsert(index_name="mel_np", ids=[str(self.id)], vectors=[self.vector])

# class mel_img(models.Model):
#     # Pinecone 벡터 필드 추가
#     id = models.AutoField(primary_key=True)
#     vector = models.JSONField(null=True, blank=True)
#     updated_at = models.DateTimeField(auto_now_add=True)
    
#     def save_to_pinecone(self):
#         # Pinecone에 벡터 색인
#         pinecone.create_index(index_name="mel_img", dimension=331)
#         pinecone.upsert(index_name="mel_img", ids=[str(self.id)], vectors=[self.vector])

# 텍스트 테이블 1. 메일, 문자
class Text_mail(models.Model):
    id = models.AutoField(primary_key=True)
    # filename = models.CharField(max_length=50, null=True)
    message = models.TextField()
    label = models.BooleanField(null=True) #default 설정 가능
    phone_number = models.CharField(max_length=50, null=True)
    message_embedding = models.TextField(null=True)

# 텍스트 테이블 2. KorCCVi
class Text_KorCCVi(models.Model):
    id = models.AutoField(primary_key=True)
    transcript = models.TextField()
    label = models.BooleanField(null=True)

# 음성 파형
# class Wave(models.Model):
#     id = models.AutoField(primary_key=True)
#     spectrogram = models.BigIntegerField()
# # 번호 1. 전화번호
# class Phone_numbers(models.Model):
#     id = models.AutoField(primary_key=True)
#     phone_number = models.CharField(max_length=50)
#     search_cnt = models.IntegerField()

# # 번호 2. 계좌번호
# class Account_numbers(models.Model):
#     id = models.AutoField(primary_key=True)
#     account_number = models.CharField(max_length=50)
#     search_cnt = models.IntegerField()

# ################
# class Sentence(models.Model):
#     text = models.CharField(max_length=255)
#     embedding = models.TextField()


