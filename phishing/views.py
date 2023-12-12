from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import Organizations, CallHistory, Text_mail, MelSpectrograms, MFCC
from django.core.paginator import Paginator

from django.views import View
from django.http import JsonResponse

from .utils import load_kcbert_model, calculate_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .forms import AudioDataForm

##### 테스트 페이지
def guide_test(request):
    return render(request, 'guide_test.html',)

# def test(request):
#     return render(request, 'home_try.html')

# Create your views here.
def home(request):
    return render(request, 'home.html')

# 신고하기
def notify(request):
    return render(request, 'notify.html')

# 번호 조회 결과 : 위험도 높음
def result_num_high(request):
    return render(request, 'result_num_high.html')

# 번호 조회 결과 : 위험도 낮음
def result_num_low(request):
    return render(request, 'result_num_low.html')

# 모델 결과 : 위험도 높음
def result_high(request):
    is_blocking_active = False
    return render(request, 'result_model_high.html', {'is_blocking_active':is_blocking_active})

# 모델 결과 : 위험도 낮음
def result_low(request):
    return render(request, 'result_model_low.html')

# 기관 정보
def rel_org(request): #기관 전체
    # page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    # paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    # page_obj = paginator.get_page(page)
    return render(request, 'rel_organization.html', {"orgs_list":orgs_list})

def financial(request): #금융 기관
    # page = request.GET.get('page', '1')
    orgs_list = Organizations.objects.all()
    # paginator = Paginator(orgs_list, 8) #페이지당 8개씩 보여주기
    # page_obj = paginator.get_page(page)
    # context = {'question_list':page_obj}
    return render(request, 'financial.html', {'orgs_list':orgs_list})

def investigative(request): #수사 및 신고기관
    orgs_list = Organizations.objects.all()
    return render(request, 'investigative.html', {'orgs_list':orgs_list})

# 대응방법 안내
def victim_guide(request):
    return render(request, 'victim_guide.html')

# # 실시간 탐지 페이지
# def real_time_detectoin(request):
#     return render(request, 'real_time_detection.html')

# 정밀검사 페이지
def text_detection(request):
    return render(request, 'text_detection.html')

# 번호 조회 페이지
# def number_search(request):
#     search_number = '' # 기본값 설정
    
#     if request.method == "POST":
#         search_number = request.POST.get('search_number', '')

#         # 데이터베이스에서 조회
#         phone_numbers = Phone_numbers.objects.filter(phone_number=search_number)
#         account_numbers = Account_numbers.objects.filter(account_number=search_number)

#         # 결과에 따라 다른 템플릿으로 렌더링
#         # 전화번호에 있으면
#         if phone_numbers.exists():
#             phone_numbers_obj = phone_numbers.first()
#             phone_numbers_obj.search_cnt += 1
#             phone_numbers_obj.save()
#             return render(request, 'result_num_high.html', {'phone_numbers': phone_numbers, 'account_numbers': account_numbers, 'search_number': search_number})
#         # 계좌번호에 있으면
#         elif account_numbers.exists():
#            account_numbers_obj = account_numbers.first()
#            account_numbers_obj.search_cnt += 1
#            account_numbers_obj.save()
#            return render(request, 'result_num_high.html', {'phone_numbers': phone_numbers, 'account_numbers': account_numbers, 'search_number': search_number})
#         else:
#             return render(request, 'result_num_low.html', {'search_number': search_number})
        
#     return render(request, 'number_search.html', {'search_number': search_number})


# 실시간 탐지 시 정보 제공 동의 여부 확인
def agreement(request):
    return render(request, 'agreement.html')

# 탐지 내역 조회테스트
def detection_history(request):
    detection_list = CallHistory.objects.all()
    return render(request, 'det_history_test.html', {'detection_list':detection_list})

###### 실시간 데이터베이스 테스트
def update_database(image_path):
    CallHistory.objects.create(image=image_path)

def my_view(request):
    # 실시간으로 멜 스펙트로그램 이미지 경로를 받아와 데이터베이스에 저장
    # 모델과 연동해서 멜 스펙트로그램 저장 및 해당 경로 받아오는 수정 필요
    image_path = "spectrograms" 
    update_database(image_path)
    return render(request, 'deepvoice_detection.html')

# 딥보이스 탐지 페이지
def deepvoice_detection(request):
    return render(request, 'deepvoice_detection.html')

########################################## KcBERT
import pandas as pd
from django.shortcuts import render
import torch
import torch.nn.functional as F
import ast
import json
# CSV 파일에서 데이터를 읽어옴
data = pd.read_csv('phishing\KcBERT_Input_test.csv')
# 캐시용 딕셔너리 생성
embedding_cache = {}
class SimilarityView(View):
    template_name = 'model_test.html'
    def get(self, request):
        return render(request, 'real_time_detection.html') #model_test.html
    def post(self, request):
        similarity_scores = []
        similarity_threshold = 0.7  # 유사도 판단 기준값
        transcript = data['Input_data']
        input_text = data['Text']
        input_num = data['Phone_number']
        all_text_mails = Text_mail.objects.all()
        # text_embedding = Text_mail.objects.values('message_embedding')
        # print(text_embedding[:1])
        # print(transcript)
        # print(input_text)
        # print(input_num)
        transcript = transcript.values[0]
        input_text = input_text.values[0]
        input_num = input_num.values[0]
        # print()
        # print(transcript)
        # print(input_text)
        # print(input_num)

        # 테스트 말고 실제 운영 가정하는 경우 input data가 text로 입력 들어왔을 때를 고려해 임베딩 진행 과정 필요할 듯

        input_data = []

        for i in all_text_mails[:50]:
            # print(i.phone_number)
            db_value = np.array(ast.literal_eval(i.message_embedding[7:-1])).reshape(-1, 1)
            # print(np.array(ast.literal_eval(i.message_embedding)[7:-1]))
            np_value = np.array(eval(transcript[7:-1]))

            similarity_score = np.dot(np_value, db_value) / (np.linalg.norm(np_value) * np.linalg.norm(db_value))
            # print(similarity_score.shape)

            if similarity_score > similarity_threshold:
                similarity_scores.append({
                    'text': i.message,
                    # 'text': transcript, 
                    # 'similarity': similarity_score
                    'similarity': similarity_score[0][0] # 행렬연산시 변경되어야 함.

                    })
                # input data 중복없이 추가하기
                if input_text not in [item['text'] for item in input_data] and input_num not in [item['phone_number'] for item in input_data]:
                    input_data.append({
                        'text':input_text,
                        'phone_number':input_num
                    })
        
        # input_data = json.dumps(input_data, ensure_ascii=False)
        result_cnt = len(input_data) #threshold 넘는 input data 개수 = 의심건수
        cat = "실시간"

        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True) #유사도 높은 순으로 정렬
        top_similarity_scores = similarity_scores[:10]
        # print(type(similarity_scores))
        print(len(input_data))
        print('input data :', input_data)
        print("Top 10 Similarities : ", top_similarity_scores)
        
        # input data만 보여주기
        # return JsonResponse({'inputs':input_data},  safe=False, json_dumps_params={'ensure_ascii': False})
        # 유사도 결과만 보여주기
        # return JsonResponse({'similarities': top_similarity_scores},  safe=False, json_dumps_params={'ensure_ascii': False})

        # 결과에 따라 다른 페이지로 연결
        # input data가 유사도 기준값을 넘는 경우
        if similarity_scores:
            result_data = [{'text':item['text'], 'sim':item['similarity']} for item in top_similarity_scores]
            return render(request, 'result_model_high.html', {'inputs':input_data, 'result_cnt':result_cnt, 'cat':cat, 'similarities':top_similarity_scores})
        # input data가 유사도 기준값을 넘지 않는 경우
        else:
            return render(request, 'result_model_low.html', {'inputs':input_data, 'cat':cat})
        
# 실시간 탐지 페이지
# SimilarityView를 실시간 탐지 페이지 html에 바로 연결하면 돼서 이 부분은 필요 없을 듯
# 버튼 클릭 없이 페이지를 로드했을 때 POST 요청을 보내 실행하고 싶다면 사용 고려해보기
# def real_time_detectoin(request):
#     rt_detection = SimilarityView() #SimilarityView 클래스의 인스턴스를 생성
#     return render(request, 'real_time_detection.html', {'similarity_data':rt_detection})
    # return rt_detection.post(request)
    # SimilarityView 인스턴스의 post 메서드를 호출 -> SimilarityView의 POST 요청 처리 로직이 실행
    # post 메서드가 반환한 결과는 return rt_detection.post(request)를 통해 real_time_detection 함수의 반환 값이 됨


##### 테스트 페이지
def test_result(request):
    return render(request, 'ppt_result.html')

from .utils import convert_audio_to_mel_spectrogram

# 오디오 파일 입력 받기
def audio_test(request):
    if request.method == 'POST':
        form = AudioDataForm(request.POST, request.FILES) # 
        if form.is_valid():
            audio_data = form.save()
            # audio_data = 'C:/Users/03123/dvdvdeep_test/bopibopi/1.wav'

            # 디버깅 출력 추가
            # print('Uploaded file path:', audio_data.audio_file.path)

            # 음성 파일을 멜 스펙트로그램으로 변환
            mel_spectrogram_img_path, mel_spectrogram_np_path, mfcc_path = convert_audio_to_mel_spectrogram(audio_data.audio_file.path)

            # 모델 인스턴스 생성 및 모델에 멜 스펙트로그램 경로 데이터 저장
            mel_spectrograms = MelSpectrograms()
            mel_spectrograms.mel_img = mel_spectrogram_img_path
            mel_spectrograms.mel_np = mel_spectrogram_np_path
            mel_spectrograms.save()

            mfcc_model = MFCC()
            mfcc_model.mfcc = mfcc_path
            mfcc_model.save()

            # return redirect('success')
            return render(request, 'audio_test.html', {'form':form})
        
    else: # 저장에 실패할 경우
        form = AudioDataForm()
    
    return render(request, 'audio_test.html', {'form':form})

def success_view(request):
    return render(request, 'success.html')

# # 버튼 누르면 mel-spectrogram 시행되게
# class SpectrogramView(View):
#     template_name = 'audio_test.html'
#     def get(self, request):
#         return render(request, 'audio_test.html')
#     def post(self, request):
#         audio_data = '1.wav'

#         # 음성 파일을 멜 스펙트로그램으로 변환
#         mel_spectrogram_img_path, mel_spectrogram_np_path = convert_audio_to_mel_spectrogram(audio_data.audio_file.path)

#         # 멜 스펙트로그램 경로를 모델에 저장
#         audio_data.mel_spectrogram_img = mel_spectrogram_img_path
#         audio_data.mel_spectrogram_np = mel_spectrogram_np_path
#         audio_data.save()

#         return render(request, 'success.html')

# import os
# import librosa
# import matplotlib.pyplot as plt
# import numpy as np
# from django.conf import settings
# from django.urls import reverse

# def audio_test(request):
#     if request.method == 'POST':
#         form = AudioDataForm(request.POST, request.FILES)
#         if form.is_valid():
#             audio_instance = form.save()

#             # Get the uploaded file path
#             audio_file_path = os.path.join(settings.MEDIA_ROOT, str(audio_instance.audio_file))

#             # Convert audio to mel spectrogram
#             y, sr = librosa.load(audio_file_path)
#             mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
#             librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max))
#             plt.savefig(os.path.join(settings.MEDIA_ROOT, 'mel_spectrogram.png'))

#             # return redirect('phishing:audio_test')
#             # return redirect(reverse('audio_test'))  # Redirect to the same page after processing
#             return render(request, 'audio_test.html', {'form':form})
#     else:
#         form = AudioDataForm()

#     return render(request, 'audio_test.html', {'form': form})



######################################################
# pinecone

# import pinecone

# # Create an index
# # "dimension" needs to be same as dimensions of the vectors you upsert
# pinecone.create_index(index_name="products", dimension=1536)

# # Connect to the index
# index = pinecone.Index(index_name="products")

# # Mock vector and metadata objects (you would bring your own)
# vector = [0.010, 2.34,...] # len(vector) = 1536
# metadata = {'text': "Approximate nearest neighbor search (ANNS) is a fundamental building block in information retrieval"}

# # Upsert your vector(s)
# index.upsert((id='some_id', values=vector, metadata=metadata)) # (id, vector, metadata)        

# ####

# #upload data
# import torch
# from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util

# import pandas as pd
# import numpy as np

# from tqdm import tqdm

# #데이터 불러오기
# baemin_df = pd.read_csv("./baemin.txt", sep="\n\n", names=['text'], engine='python')
# baemin_df['text'] = baemin_df['text'].apply(lambda x: '[배민] ' + x)
# baemin_guide = baemin_df['text'].to_list()
# kakao_df = pd.read_csv("./kakao.txt", sep="\n\n", names=['text'], engine='python')
# kakao_df['text'] = kakao_df['text'].apply(lambda x: '[카톡] ' + x)
# kakao_guide = kakao_df['text'].to_list()

# guide_li = baemin_guide + kakao_guide

# embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")
# guide_embedding_li = []

# for idx, guide in enumerate(guide_li):
#    guide_embedding_li.append((str(idx), embedder.encode(guide, convert_to_tensor=True).tolist(), {"info":guide.split(":")[0]}))

# import pinecone      

# pinecone.init(      
# 	api_key='api-key',      
# 	environment='gcp-starter'      
# )      
# index = pinecone.Index('stepstones-db')

# index.upsert(guide_embedding_li)



# ############
# #query data
# import torch
# from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util

# import pandas as pd
# import numpy as np

# from tqdm import tqdm


# baemin_df = pd.read_csv("./baemin.txt", sep="\n\n", names=['text'], engine='python')
# baemin_df['text'] = baemin_df['text'].apply(lambda x: '[배민] ' + x)
# baemin_guide = baemin_df['text'].to_list()
# kakao_df = pd.read_csv("./kakao.txt", sep="\n\n", names=['text'], engine='python')
# kakao_df['text'] = kakao_df['text'].apply(lambda x: '[카톡] ' + x)
# kakao_guide = kakao_df['text'].to_list()

# guide_li = baemin_guide + kakao_guide

# pinecone.init(      
# 	api_key='api-key',      
# 	environment='gcp-starter'      
# )      
# index = pinecone.Index('stepstones-db')


# # def get_sim_topk(query_text, k):
# #   query_em = embedder.encode(query_text, convert_to_tensor=True).tolist()

# # 가장 유사한 내용 k개 찾기
#   result = index.query( # 벡터 DB에서 찾아줌 _ 인덱스를
#     vector=query_em,
#     top_k=k,
#     include_values=True

#   ).matches
# #   return [guide_li[int(re.id)] for re in result]
  
# get_sim_topk("로그인 하는 방법을 모르겠어", 5)