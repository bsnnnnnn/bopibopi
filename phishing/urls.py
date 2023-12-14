from django.urls import path
from . import views

app_name = 'phishing'

urlpatterns = [
    path('', views.home, name='home'),
    path('notify/', views.notify, name='notify'), #신고 페이지
  
    path('result_high/', views.result_high, name='result_high'), #모델 결과 페이지 : 위험도 높음
    path('result_low/', views.result_low, name='result_low'), #모델 결과 페이지 : 위험도 낮음
    path('rel_org/', views.rel_org, name='rel_org'), #관련 기관 리스트
    path('financial/', views.financial, name='financial'), #금융기관 리스트
    path('investigative/', views.investigative, name='investigative'), #수사 및 신고기관 리스트
    path('victim_guide/', views.victim_guide, name='victim_guide'), #대응방법 가이드
    path('agreement/', views.agreement),
    
    path('text_detection/', views.text_detection, name='text_detection'), #정밀검사 페이지

    
    # 테스트 페이지
    path('deepvoice_detection/', views.deepvoice_detection, name='deepvoice_detection' ), # 딥보이스 탐지 페이지
    path('detection_history/', views.detection_history, name='detection_history' ),
    # path('model_test/', views.SimilarityView.as_view(), name='model_test'),
    path('real_time_detection/', views.SimilarityView.as_view(), name='real_time_detection'), # KcBERT : 실시간 탐지 페이지
    # path('guide_test/', views.guide_test, name='guide_test' ), 

    path('test_result/', views.test_result, name='test_result' ), #피피티용 결과 페이지

    # 오디오 파일 입력 및 멜 스펙트로그램 테스트 페이지
    # path('audio_test/', views.audio_test, name='audio_test'),
    # path('audio_test/', views.SpectrogramView.as_view(), name='audio_test'),
    path('success/', views.success_view, name='success'),


    # path('real_time_detection/', views.real_time_detectoin, name='real_time_detection'), #실시간 탐지 페이지
    # path('number_search/', views.number_search, name='number_search'), #의심번호 입력
    # path('result_num_high/', views.result_num_high, name='result_num_high'), #번호 조회 결과 페이지 : 위험도 높음
    # path('result_num_low/', views.result_num_low, name='result_num_low'), #번호 조회 결과 페이지 : 위험도 낮음
]
