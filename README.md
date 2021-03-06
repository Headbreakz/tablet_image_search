

```bash
└─tablet_image_search
    │  .gitignore
    │  README.md
    │
    ├─App
    │      main.py
    │
    ├─Code
    │      App_code.ipynb
    │      tablet_classify.ipynb
    │      googleAPI.ipynb
    │      image_save.ipynb
    │      image_search.ipynb
    │      labels.txt
    │      model_study.ipynb
    │      test.ipynb
    │      모양.txt
    │      색상.txt
    │
    └─Data
           공공데이터개방_낱알식별목록.csv
           공공데이터개방_낱알식별목록.xlsx
           공공데이터개방_낱알식별목록_re.csv
           공공데이터개방_낱알식별목록_re.xlsx
    
```

![알약알리미](https://github.com/Headbreakz/tablet_image_search/blob/master/image/main.jpg)





### 알약 사진을 통한 알약 정보 알리미

http://pf.kakao.com/_nxkxbaxb

---

1. 프로젝트 배경

   * 세계 의약 시장은 17년 306조원에서 23년 500조의 규모로 확대 될 전망
   * 19년 기준 우리나라 의약품 부작용 보고 건수 총 13680건
   * 상비약의 경우 포장되어 있지만, 처방 받은 약의 경우 약 봉투 외에 정보를 확인 할 곳은 3개의 사이트(약학정보원,네이버 건강의약품,의약 도서관)과 3개의 APP 서비스
   * 한정된 알약 검색 서비스 – 스스로 정보를 입력

   

2. 프로젝트 목표

   * 알약 이미지 분류 model 정확도 90% 이상 
   * 알약 서비스 정확도 80% 

   

3. 프로젝트 범위

   1. 데이터 수집

      1. 의약품안전나라- 알약 데이터 수집

   2. ML 모델 구현

      1. Python 3.6 사용
      2. 알약 이미지 분류 모델 - one shot algorithm 활용
      3. 알약 글씨 추출기 - 미정

   3. 서비스 배포

      1. 카카오 플러스 친구 - kakao open builder 사용

         

4. 프로젝트 계획 ( 총 3 주 계획 )

   | 순번  |        활동명         |     시작일     |     종료일     |               주요 활동 내용               |
   | :---: | :-------------------: | :------------: | :------------: | :----------------------------------------: |
   |   1   |     아이디어 선정     |   2020.04.08   |   2020.04.09   |           아이디어 선정 및 검토            |
   |   2   |     서비스 스케치     |   2020.04.10   |   2020.04.10   |             서비스 스케치 작성             |
   |   3   | 데이터 수집 및 전처리 |   2020.04.10   |   2020.04.14   |           데이터 수집 후 전처리            |
   |   4   |   이미지 분류 model   |   2020.04.13   |   2020.04.17   |           알약 이미지 분류 model           |
   |   5   |      이미지 추출      |   2020.04.15   |   2020.04.17   |     알약 이미지 추출 API 탐색 및 적용      |
   | ~~6~~ |      ~~DB 구축~~      | ~~2020.04.20~~ | ~~2020.04.20~~ |      ~~알약 정보 DB에 저장(SQLite)~~       |
   |   7   |      서비스 제작      |   2020.04.20   |   2020.04.22   | 카카오 플러스 친구 제작 및 분류 model 적용 |
   |   8   |     서비스 디자인     |   2020.04.23   |   2020.04.23   |   카카오 플러스 친구 디자인 제작 및 적용   |
   |   9   |      서비스 test      |   2020.04.24   |   2020.04.24   |              서비스 전체 test              |

   

   

---

#### 전체 진행과정

* 1일차 /0408
  * 주제 재확인 
  
  * 작성했던 기획안 확인
  
  * 기존 알약 정보 사이트
    * 약학정보원 (https://www.health.kr/searchIdentity/search.asp)
    * 의약품 식별 표시 (http://www.pharm.or.kr/search/drugidfy/search.asp)
    * Druginfo(https://www.druginfo.co.kr/identy/identy_char.aspx)
    * 의약품안전나라(https://nedrug.mfds.go.kr/searchDrug)
    
  * 기존 알약 정보 App
    
      * 약학정보원 의약품 검색(https://apps.apple.com/kr/app/%EC%95%BD%ED%95%99%EC%A0%95%EB%B3%B4%EC%9B%90-%EC%9D%98%EC%95%BD%ED%92%88%EA%B2%80%EC%83%89/id456851567)
      
  * 진행된 프로젝트
  
      http://www.ndsl.kr/ndsl/commons/util/ndslOriginalView.do?dbt=TRKO&cn=TRKO201800005241&rn=&url=&pageCode=PG18
  
* 2일차 /0409
  
  * 1일차 진행과정 자세하게 내용 추가
  * 일정 계획표 작성
  * 알약 사진 저장 code 완성
  
* 3일차 /0410
  * 알약 사진 저장 (약 2만2천개)
  * 알약 사진 google drive 저장 - 4일차
  * 서비스 스케치
  * 주제 발표 자료 작성
  
* 4일차 /0413
  
  * 주제 발표
    
    * 발표 후 의견
    
    1. 알약의 생김새가 형태와 색이 거의 비슷한테 유의미한 분류가 가능한가요
    2. 어르신들을 위한 음성 서비스
    
  * 알약 사진 google drive 저장 (1000개)
  
  * 알약 분류 model 알고리즘 검색
  
    * One shot learning with Siamese Networks using keras
  
* 5일차 /0414
  * One shot learning with Siamese Networks using keras 을 이용하여 model 구현 - 실패
  * 현재 카테고리화 되어 있지 않아서, 분류하기에 어려움
  * 카테고리로 묶기 위해, 분류 번호를 사용할 예정
  
* 6일차 /0416
  * model에 대한 전체 수정
  * 데이터 이미지와 색깔정보, 모양 정보를 input하여 model를 만들기 위해 데이터 전처리
  * 알약 사진의 배경을 없애기 위해 GrabCut을 사용하여, 배경 부분을 검정색으로 표시
  * 현재 jpg 파일 형식을 PNG 형식으로 전환하여, RGBA로 변환
  * 검정 + 남아 있는 부분에 대한 색상을 (0,0,0,0)으로 변환
  * keras Multi input 모델 검색 
  
* 7일차 /0417

  * 모델 수정

    * 다중 라벨 분류 (Multi - label classification with keras)
    * 이미지 1개 
    * 라벨 2개 ( 색상, 모양 )
  * 참고 링크
    * [케라스를 이용한 다중 라벨 분류](https://github.com/KerasKorea/KEKOxTutorial/blob/master/35)
  * 알약 사진 재 저장 -22,878개 145 목록
  * 알약 사진을 형태와 색상으로 나눠서 저장 
  
* 8일차 /0418-19 추가활동

  * 배경 제거 작업 - grabcut
    * 배경 제거 작업에서 기타로 분류된 알약 제외- 현재 22,400개 131목록
    * 기타 알약사진 제거
  * 검정색 배경을 투명한색으로 변경
    * 변경 작업에서 노이즈 이미지 제거 - 824개 제거 - 현재 총 21,603개 126목록
  
* 9일차 /0420

  * Colab에서 AWS로 이동하여 model 학습 
    * 이동 이유 : Colab 메모리 부족
  * 카카오 플러스 친구 생성
  * 최종 알약 추출 code 진행중
  * 추가 작업을 통해 model 학습 완료
  
* 10일차 /0421

  * 카카오 플러스 친구 생성 - 알약 알리미
  * 카카오 플러스 친구에서 code가 작동 여부 확인
  * 최종 code 수정 필요
  
* 11일차 /0422

  * python을 json으로 수정
  * 쳇봇 Timeout 문제 해결해야함 - 해결
  * 추가 작업을 통해 완성 
  
* 12일차 /0423

  * 전체 TEST 진행 및 오류 확인 중
  * 여러장의 사진 중 맨 처음만 인식하도록 수정
  * 알약 알리미 사용법 작성
  * 발표 PPT 작성
  * model 정확도 및 오버피팅 해결을 위해 재조정

---

#### 어려운점, 잘못된점, 못한점

* 전체 알약에 대해서 Y을 이름 값으로 하여, One shot learning with Siamese Networks using keras을 한다면 될거라고 생각함
  * 2만개가 넘는 클래스로 인해 어려움 봉착
* keras functional API를 통해 사진과 모양,색상정보를 따로 모델링하여 진행하면 될거라고생각
  * 사진과 모양,색상이 같이 매치되지 못함- 1번사진과 1번 모양,색상 정보로 차례대로 입력되지 않음
  * 결국 따로따로 input들이 각 각의 모델을 통해 나중에 합쳐지는 모델이 되어서, 맞지 않다고 판단
* 모양,색상으로 분류
  * 진행된 프로젝트를 참고 사항으로 모양,색상으로 분류하기로함
  * 처음부터 데이터를 다시 정리해야함
  * 한글 파일로 폴더명을 할 경우, python에서 오류 - 영어로 전환 후 다시 데이터 정리
* keras,tensorflow version이 중요
* 챗봇의 Timeout을 고려하여, 제작해야 함
  * 5초 제한에 걸려서, 2부분으로 분할하여 진행
* Json의 구조를 정확하게 파악해야 함



---

#### 알약 알리미 사용법

http://pf.kakao.com/_nxkxbaxb/50765274