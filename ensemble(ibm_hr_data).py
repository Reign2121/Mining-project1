import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

hr = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Data/WA_Fn-UseC_-HR-Employee-Attrition 2.csv")

hr = hr.astype({'JobLevel': 'category', 'StockOptionLevel': 'category', 'Education':'category',
                'MaritalStatus':'category','Attrition':'category', 'OverTime':'category', 'Gender':'category',
                'BusinessTravel':'category', 'Department':'category','EducationField':'category','JobRole':'category'
                })
#변수 제거(모두 같은 값만 가지므로
hr.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis=1,inplace=True)

#의사결정나무는 정규화 불필요
hr_copy = hr.copy().drop('Attrition',axis=1)
num_fea = [column for column in hr_copy.columns if hr_copy[column].dtype != "category"]
cat_fea = [column for column in hr_copy.columns if hr_copy[column].dtype == "category"]

#원핫인코더로 더미화, get_dummies도 가능

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

dummies = ohe.fit_transform(hr[cat_fea]).toarray()

df_dummies = pd.DataFrame(dummies, columns = ohe.get_feature_names(hr[cat_fea].columns))

hr_dum=pd.concat([hr_copy[num_fea].reset_index(), df_dummies], axis=1)
hr_dum.drop('index',inplace=True,axis=1)

hr_dum.columns

#set 분리
x = hr_dum
y = hr.Attrition
from imblearn.under_sampling import *
# oss = 토멕랭크 + cnn <채택>
X_samp, y_samp = OneSidedSelection().fit_resample(x,y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_samp, y_samp, test_size=.2, random_state = 40) #,shuffle = True, stratify = y_samp

from sklearn.tree import DecisionTreeClassifier
#성장제한 / 파라미터 튜닝
max_depth = [4,5,6,7,8,9,10]
for i in max_depth:
  dt_md = DecisionTreeClassifier(min_samples_leaf = 20
                                 , criterion = "entropy",max_depth = i, class_weight= 'balanced',random_state = 4)
  dt_md.fit(x_train,y_train)
  print("max_depth:",i)
  print('정확도:',dt_md.score(x_test,y_test))

#가중치 줬을때
  dt_5 = DecisionTreeClassifier(min_samples_leaf = 20,
                                #말단노드최소수  
                                criterion = "entropy",
                                #엔트로피 지수 사용
                                max_depth = 5,
                                #최대성장 
                                class_weight = 'balanced',
                                #가중치 부여
                                random_state=5)
                                
  dt_5.fit(x_train,y_train)
  print(dt_5.score(x_test,y_test)) #mean accuracy

#from sklearn.model_selection import cross_val_score

#svm
from sklearn.svm import SVC

svm = SVC(C=0.1, gamma='auto', probability=True, random_state = 5)
svm.fit(x_train,y_train)
svm.score(x_test,y_test) #mean accuracy

#svm_cv=cross_val_score(svm,x_test,y_test,cv=10,scoring='accuracy')
#svm_cv

#로지스틱 회귀
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state=5)
log.fit(x_train,y_train)
log.score(x_test,y_test) #mean accuracy

#log_cv=cross_val_score(log,x_test,y_test,cv=10,scoring='accuracy')
#log_cv

"""# **Voting with svm, log, dt**"""

from sklearn.ensemble import VotingClassifier

vc= VotingClassifier(estimators=[("DT", dt_5),("SVM", svm), ("LR", log)], voting = "soft")
vc.fit(x_train,y_train)
vc.score(x_test,y_test) #same as svm #mean accuracy

"""# **Bagging (random forest)**"""

#랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(class_weight='balanced', criterion = "entropy", random_state = 25)
rf.fit(x_train,y_train)
rf.score(x_test,y_test) #mean accuracy

print(rf.feature_importances_)

rf_importance = pd.DataFrame()
rf_importance['Feature'] = x_train.columns # 설명변수 이름
rf_importance['Importance'] = rf.feature_importances_ # 설명변수 중요도 산출

# 변수 중요도 내림차순 정렬
rf_importance.sort_values("Importance", ascending = False, inplace = True)
print(rf_importance.round(3))
# 변수 중요도 오름차순 정렬
#rf_importance.sort_values("Importance", ascending = True, inplace = True)

# 변수 중요도 시각화
coordinates = range(len(rf_importance)) # 설명변수 개수만큼 bar 시각화

plt.figure(figsize=(10, 15))# 크기 조정은 위에서
plt.barh(y = coordinates, width = rf_importance["Importance"])
plt.yticks(coordinates, rf_importance["Feature"]) # y축 눈금별 설명변수 이름 기입
plt.xlabel("Feature Importance") # x축 이름
plt.ylabel("Features") # y축 이름
plt.show()
#plt.savefig('../figure/' + 'random_forest' + '_feature_importance.png') # 변수 중요도 그래프 저장

"""# **Boosting(xg_boost)**"""

#pip install xgboost

from xgboost import XGBRegressor, XGBClassifier

xgb = XGBClassifier(random_state=1)
xgb.fit(x_train, y_train)
xgb.score(x_test,y_test) #mean accuracy

"""accuracy_score"""

from sklearn.metrics import accuracy_score

ensembles = [vc, xgb, rf]

for ensemble in ensembles:
    pred = ensemble.predict(x_test)
    name = ensemble.__class__.__name__
    print(f"{name} 테스트 정확도 : {accuracy_score(y_test, pred)}") #not mean!!!!

"""# DT 성능"""

y_predict_dt = dt_5.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_dt, target_names = ["퇴사x","퇴사"]))

"""# RF 성능"""

y_predict_rf = rf.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_rf, target_names = ["퇴사x","퇴사"]))

"""# XGB 성능"""

y_predict_xgb = xgb.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict_xgb, target_names = ["퇴사x","퇴사"]))