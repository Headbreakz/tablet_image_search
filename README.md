### 알약 사진을 통한 알약 정보 알리미

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

   * 1 ~ 3 일
     * 데이터 수집 및 데이터 전처리
   * 3 ~ 9 일
     * 이미지 분류 model 및 학습
     * 알약 글씨 추출을 위한 API 탐색 및 적용
   * 10 ~ 13 일
     * 카카오 플친 확인 - 쳇봇 구조,
   * 14 일
     * 카카오 플친 디자인
   * 15 일
     * 전체 test , 작동 및 오류 확인

   

   

---

#### 전체 진행과정

* 1일차 

  * 아이디어 재확인 

  * 작성했던 기획안 확인

  * 타 서비스 및 데이터 제공 사이트 확인

  * 데이터(이미지) 수집 code 

    