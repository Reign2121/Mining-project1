# Mining-project

Data Mining using Decision Tree and codes 

주제: 조직관리방안의 일환으로, 분류 문제에 적합한 알고리즘들을 이용해 이탈(퇴사)모델을 만들어 시사점을 제시한다.

데이터 : Human Resource data in kaggle made by ibm data scientist

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?datasetId=1067&searchQuery=deci

label: "attrition"

__________

우선 저는 탐색적 데이터 분석(eda)에 있어서 "데이터를 잘 살펴볼 수 있는 코드를 디자인" 하는 것에 노력했습니다.

이를 통해 레이블의 균형이 맞지 않는 것을 발견하여 이를 보완하기 위해 언더 샘플링 기법을 적용했습니다.

제가 사용한 oss는 이 방법은 마찬가지로 언더샘플링 기법들인 토맥랭크와 cnn기법을 합친 기법입니다.

- 토멕랭크는 분포가 작은 클래스의 데이터에서 가장 가까운 다른 클래스의 데이터를 찾아서 죽이는 기법입니다. 이렇게 되면, 중심분포는 유지하면서 이상치나 분류기준 선에 붙어있는 데이터들을 제거힐 수 있습니다.

- cnn 기법은 분포가 큰 클래스에서 소수 클래스와 지나치게 먼 거리의 샘플을 제거하는 방법입니다. oss는 이 둘을 적절히 섞은 샘플링 기법입니다. 

저는 이 데이터에서 소수 클래스의 수가 250개가 채 되지 않아서 단순 랜덤샘플링으로 1450개에 달하는 다수 데이터를 무작위로 줄이면 데이터의 왜곡이 크게 생길 수 있다고 보았습니다.

