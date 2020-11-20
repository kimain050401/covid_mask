# covid_mask
## 주의 및 안내사항
***본 코드는 단양소백산중학교에서 활용할 "마스크 미착용 단속 카메라" 코드입니다.  
아직 개발단계에 있으며, 본 코드를 사용함에 따른 불이익은 책임지지 않습니다.***
## 개발 취지
학생들이 마스크를 올바르게 착용하지 않고 다니고, 마스크 의무화가 진행됨에 따라 효과적으로 학생들이 마스크를 착용하게 하기 위해서 제작되었습니다.
## 날짜별 업데이트 내역
* [2020.11.20.] 메인 코드 커밋 완료. dlib 라이브러리 불안정. 얼굴인식 너무 잘됨(마스크 써도 인식됨).
## 교내 단속 예정 구역
* 두개의 열화상 카메라 옆 컴퓨터(웹캠 장착되어 있음)(1층 본관 추출입구 & 급식실 앞)
* 계단 중간지점(C교무실 부분 계단 & 보건실 부분 계단)(나무계단 제외)
* 효율적인 단속(학생들이 단속 카메라 앞에서만 마스크를 쓰는 일 방지)을 위해 예고하지 않고 단속 카메라를 지속적으로 이동할 계획
## 기타 이슈 사항
* 계단 중간지점 설치 시 전원 및 인터넷 공급 방법
* dlib 라이브러리 설치 안됨(아나콘다로 설치해야하지만 컴퓨터마다 컴바컴)
## To-Do List
* ~opencv 질문한것 답장왔는지 확인~
* ~학교 컴퓨터에서 웹캠을 연결해서 코딩~
* 코드를 실행 파일로 만들어서 열화상카메라 컴퓨터에서 실행(1층 주출입구&급식실 앞)
* dlib 라이브러리 설치 되게끔 하기
* 기존 입 검출 방식에서 이미지 데이터를 모아 머신러닝으로 검출하는 방법으로 변경
