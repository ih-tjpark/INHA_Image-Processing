# Python_ImageProcessing
인하대학교 영상처리 전공교육

### Mid Test - Moving Line Detection
- 동영상에서 움직이는 프레임을 검출해 직선을 찾은 후 직선에 라인을 그려주는 프로젝트 
![image](https://user-images.githubusercontent.com/57780594/169327438-35686785-dc85-4c07-8f9e-4b78ce9ab040.png)

<hr>


### Final Project - CarNumber Detection
- 다양한 색상과 각도의 자동차 번호판 사진에서 번호판 영역을 찾고 차 번호를 검출하는 프로젝트

| 1. Original Image | 2. Preprocessing|
| ------------------| ---------------- |   
|![1](https://user-images.githubusercontent.com/57780594/169329057-806377bf-4c77-41c7-b311-f4e695dbcb91.jpg)|![2](https://user-images.githubusercontent.com/57780594/169330398-f3d208f4-eb95-4156-945f-44823dfd33f5.jpg)|

| 3. Top30 Contour | 4. Possible Contour|
| ------------------| ---------------- |  
|![3](https://user-images.githubusercontent.com/57780594/169334210-c7ab82ee-c0dc-418d-bf8d-8421d2006bbe.jpg)|![4](https://user-images.githubusercontent.com/57780594/169334617-2cf39f18-d7ff-4d06-9f48-17413d93c34a.jpg)|

| 5. Match Contour | 6. Rotate Image|
| ------------------| ---------------- | 
|![5](https://user-images.githubusercontent.com/57780594/169335641-78576c66-adcb-4e30-a375-7d98e6658800.jpg)|![r](https://user-images.githubusercontent.com/57780594/169335663-255ebfef-6043-459a-92c1-cd0d6c16ca8c.jpg)|

| 7. Cropped CarNumber Image| 8. Image enlargement (Final image)|
| ------------------| ---------------- | 
|![6](https://user-images.githubusercontent.com/57780594/169335827-b1a60252-e185-41e0-a0cc-f585270deeab.jpg)|![7](https://user-images.githubusercontent.com/57780594/169336243-af568841-dbe8-4966-9c26-1ae51ffbc096.jpg)|

| 9. pytesseract OCR| 10.Repeat|
| ------------------| ---------------- | 
|![pytesseract](https://user-images.githubusercontent.com/57780594/169337845-642bcbda-83b0-41f9-bdc2-a3f2d5d49e65.png)| ![1200px-Repeat_font_awesome svg](https://user-images.githubusercontent.com/57780594/169339282-50b1385f-176a-4287-849c-1ec37b5565a0.png)|
|OCR 검출결과: 02허9757| 모든 사진 1~9번 과정 반복|

<br><br><br>
### 최종 결과  
![t](https://user-images.githubusercontent.com/57780594/169342540-188ab97d-d9ac-43ae-a158-eb81a373eaf5.jpg)
