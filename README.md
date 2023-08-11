# Mining-project

Data Mining using Decision Tree and codes 

주제: 조직관리방안의 일환으로 분류 문제에 적합한 알고리즘을 이용해 이탈(퇴사)모델을 만들어 시사점을 제시한다.

데이터 : Human Resource data in kaggle made by ibm data scientist

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?datasetId=1067&searchQuery=deci

label: "attrition" (퇴사 여부)

__________

우선 저는 탐색적 데이터 분석(eda)에 있어서 "데이터를 잘 살펴볼 수 있는 코드를 디자인" 하는 것에 노력했습니다.

이를 통해 레이블의 균형이 맞지 않는 것을 발견하여 이를 보완하기 위해 언더 샘플링 기법을 적용했습니다.

제가 사용한 oss는 이 방법은 마찬가지로 언더샘플링 기법들인 토맥랭크와 cnn기법을 합친 기법입니다.

- 토멕랭크는 분포가 작은 클래스의 데이터에서 가장 가까운 다른 클래스의 데이터를 찾아서 죽이는 기법입니다. 이렇게 되면, 중심분포는 유지하면서 분류기준 선에 붙어있는 데이터들을 제거힐 수 있습니다.

- cnn 기법은 분포가 큰 클래스에서 소수 클래스와 지나치게 먼 거리의 샘플(예컨데 이상치)을 제거하는 방법입니다. oss는 이 둘을 적절히 섞은 샘플링 기법입니다. 

저는 이 데이터에서 소수 클래스의 수가 250개가 채 되지 않아서 단순 랜덤샘플링으로 1450개에 달하는 다수 데이터를 무작위로 줄이면(수를 맞추면) 데이터의 왜곡이 크게 생길 수 있다고 보았습니다.

또한 데이터 셋(train/test) 분리에 있어서도 y에 따라 층화추출하는 방법 또한 불균형 문제를 다루는 것에 유효하지만, 이 모델에서는 오히려 성능이 하락하는 것을 발견하여 이를 제외하였습니다.

* train_test_split(x, y, test_size=.2, random_state=4) #,shuffle = True #True가 디폴트, stratify = y_samp 
__________

<img width="1206" alt="image" src="https://user-images.githubusercontent.com/121419113/217737145-f4bd5574-f405-4860-8de0-9a6ca01b37c5.png">

Insight

- 젊은 직원(32.5세 이하)의 이탈이 두드러짐. 특히 미혼인 직원들의 이탈이 많다.

- 월급이 13026 보다 많으면 대부분 이탈을 안한다. (약 58명 중 4명 정도만 퇴사)

- 반대로 월급이 2458보다 작고, daily rate이 786보다 작으면 무조건 퇴사를 하였다.(엔트로피 0)

- 근속연수 1.5년 이하에서 대부분 퇴사

- 부서 중에서 영업 부서에서 이탈이 두드러짐. 그 중 특히 직무 만족도 3.5 이하, 환경 만족도 2.5 이하인 직원들의 퇴사 비율은 2배이다.

__________

앙상블

예측성능을 향상시키기 위해 앙상블 기법 여러 개를 구현하여 성능을 비교하였다.


<img width="475" alt="image" src="https://user-images.githubusercontent.com/121419113/218019232-b873014d-86b7-42ae-b49a-91c3b841060c.png">



</br>
-보팅

with SVM, Logistic Regression, DT


</br>
-배깅 (랜덤 포레스트)

변수중요도 산출

<img width="784" alt="image" src="https://user-images.githubusercontent.com/121419113/218018621-aebbfd97-bce3-4d78-ae4b-dfe97db390f6.png">

Monthly income, 즉, 한 달 월급이 분기에 있어서 가장 중요한 변수로 선정되었다. (퇴사에 있어서 가장 중요한 요인)


</br>
-부스팅 (xgboost)

랜덤포레스트와 큰 차이가 나지 않지만 가장 높은 정확도를 보인다.

그런데, f-1 score에서는 xg부스트 모델이 랜덤포레스트에 비해 2배가 높다.

이는 클래스 불균형 상태에도 꽤 높은 성능을 보일 수 있다는 것인데, 동시에 과적합 문제를 안고 있다고도 볼 수 있다.

__________

reference: https://seollane22.tistory.com/16
