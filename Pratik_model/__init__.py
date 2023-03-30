from sklearn.metrics import r2_score,accuracy_score,mean_squared_error,mean_absolute_error,classification_report,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression,LinearRegression,PassiveAggressiveClassifier,PassiveAggressiveRegressor,Lasso,RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,VotingRegressor,HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from xgboost import XGBClassifier,XGBRegressor

import warnings
warnings.simplefilter('ignore',) 


#Classifiaction models
logreg=LogisticRegression(random_state=42)
rfc=RandomForestClassifier(random_state=42)
exc=ExtraTreesClassifier(random_state=42)
dc=DecisionTreeClassifier(random_state=42)
svc=SVC(random_state=42)
gbc=GaussianNB()
knn=KNeighborsClassifier()
pacc=PassiveAggressiveClassifier(random_state=42)
rc=RidgeClassifier(random_state=42)
lgbmc=LGBMClassifier(random_state=42)
hgbc=HistGradientBoostingClassifier(random_state=42,verbose=0)
xgbc=XGBClassifier(random_state=42,verbosity = 0)
cbc=CatBoostClassifier(random_state=42,verbose=0)





#Regression models
linereg=LinearRegression()
rfr=RandomForestRegressor(random_state=42)
ext=ExtraTreesRegressor(random_state=42)
gbr=GradientBoostingRegressor(random_state=42)
svr=SVR()
knnr=KNeighborsRegressor()
dtr=DecisionTreeRegressor(random_state=42)
pacr=PassiveAggressiveRegressor(random_state=42,average=3)
las=Lasso(random_state=42)
lgbmr=LGBMRegressor(random_state=42)
hgbr=HistGradientBoostingRegressor(random_state=42)
xgbr=XGBRegressor(random_state=42,verbosity = 0)
cbr=CatBoostRegressor(random_state=42,verbose=0)





name=[hgbc,pacc,dc,rfc,exc,logreg,rc,lgbmc,knn,svc,cbc,xgbc,gbc]
name_r=[pacr,hgbr,gbr,dtr,rfr,ext,lgbmr,las,knnr,linereg,cbr,xgbr,svr]



class smart_classifier:
    def __init__(self,x,y):
        self.__x=x
        self.__y=y

    def accuracy_score(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                print('CatBoost Classifier  - ',accuracy_score(testy,pred))
                print()

            elif i==xgbc:
                print('XGBoost Classifier - ',accuracy_score(testy,pred))
                print()
            else:   
                print( '{}     -'.format(i),accuracy_score(testy,pred))
                print()


    def f1_score(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                print('CatBoost Classifier - ',f1_score(testy,pred))
                print()

            elif i==xgbc:
                print('XgBoost Classifier - ',f1_score(testy,pred))
                print()

            else:   
                print('{}     - '.format(i),f1_score(testy,pred))
                print()
           


            
    def cross_validation(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            score=cross_val_score(i,self.__x,self.__y,cv=5)
            if i==cbc:
                print('CatBoost Classifier - ',score.mean())
                print()

            elif i==xgbc:
                print('XgBoost Classifier - ',score.mean())
                print()

            else:   
                print('{}     - '.format(i),score.mean())
                print()

        


            
    def precision_score(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        if (trainy.nunique())>2: # Multiclass
            print('It is multi-classed problem so for that we are calculating Presicion Score in average = "macro"')
            print()
            print()

            for i in name:
                i.fit(trainx,trainy)
                pred=i.predict(testx)
                if i==cbc:
                    print('CatBoost Classifier - ',precision_score(testy,pred,average='macro'))
                    print()

                elif i==xgbc:
                    print('XgBoost Classifier - ',precision_score(testy,pred,average='macro'))
                    print()

                else:   
                    print('{}    - '.format(i),precision_score(testy,pred,average='macro'))
                    print()

            

        else:
            for i in name:
                i.fit(trainx,trainy) # Binary
                pred=i.predict(testx)
                if i==cbc:
                    print('CatBoost Classifier - ',precision_score(testy,pred,average='macro'))
                    print()

                elif i==xgbc:
                    print('XgBoost Classifier - ',precision_score(testy,pred,average='macro'))
                    print()

                else:   
                    print('{}     - '.format(i),precision_score(testy,pred,average='macro'))
                    print()
                
           
        
            
            

    def recall_score(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        if (trainy.nunique())>2: # Multiclass
            print('It is multi-classed problem so for that we are calculating Recall Score in average = "macro"')
            print()
            print()

            for i in name:
                i.fit(trainx,trainy)
                pred=i.predict(testx)
                if i==cbc:
                    print('CatBoost Classifier - ',recall_score(testy,pred,average='macro'))
                    print()

                elif i==xgbc:
                    print('XgBoost Classifier - ',recall_score(testy,pred,average='macro'))
                    print()

                else:   
                    print('{}     - '.format(i),recall_score(testy,pred,average='macro'))
                    print()


        

        else:
            for i in name:
                i.fit(trainx,trainy) # Binary
                pred=i.predict(testx)
                if i==cbc:
                    print('CatBoost Classifier - ',recall_score(testy,pred,average='macro'))
                    print()

                elif i==xgbc:
                    print('XgBoost Classifier - ',recall_score(testy,pred,average='macro'))
                    print()

                else:   
                    print('{}     - '.format(i),recall_score(testy,pred,average='macro'))
                    print()
               

            
            
    def classification_report(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                print('CatBoost Classifier - ')
                print(classification_report(testy,pred),end='')
                print()
                print()

            elif i==xgbc:
                print(' XgBoost Classifier - ')
                print(classification_report(testy,pred),end='')
                print()
                print()

            else:   

                print(' {}  - '.format(i))
                print(classification_report(testy,pred),end='')
                print()
                print()
            


    def confusion_matrix(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                print('CatBoost Classifier   - ')
                print(confusion_matrix(testy,pred),end='')
                print()
                print()

            elif i==xgbc:
                print('XgBoost Classifier   - ')
                print(confusion_matrix(testy,pred),end='')
                print()
                print()

            else:   

                print(' {}  - '.format(i))
                print(confusion_matrix(testy,pred),end='')
                print()
                print()
            

            
    def mean_squared_error(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                    print('CatBoost Classifier  - ',mean_squared_error(testy,pred))
                    print()

            elif i==xgbc:
                print('XgBoost Classifier   - ',mean_squared_error(testy,pred))
                print()

            else:   
                print('{}     - '.format(i),mean_squared_error(testy,pred))
                print()
           
            
            
    def mean_absolute_error(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbc:
                    print('CatBoost Classifier   - ',mean_absolute_error(testy,pred))
                    print()

            elif i==xgbc:
                print('XgBoost Classifier    - ',mean_absolute_error(testy,pred))
                print()

            else:   
                print('{}      - '.format(i),mean_absolute_error(testy,pred))
                print()
            


            
    def overfitting(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name:
            i.fit(trainx,trainy)
            pred_train=i.predict(trainx)
            pred_test=i.predict(testx)
            if i==cbc:
                print('Training Accuracy of CatBoost Classifier    - ',accuracy_score(testy,pred_test))
                print('Testing Accuracy  of CatBoost Classifier    - ',accuracy_score(testy,pred_test))
                print()

            elif i==xgbc:
                print('Training Accuracy of XgBoost Classifier    - ',accuracy_score(testy,pred_test))
                print('Testing Accuracy  of XgBoost Classifier    - ',accuracy_score(testy,pred_test))
                print()

            else:

                print('Training Accuracy of {}     - '.format(i),accuracy_score(trainy,pred_train))
                print('Testing Accuracy  of {}     - '.format(i),accuracy_score(testy,pred_test))
                print()
           
            

class smart_regressor:
    
    def __init__(self,x,y):
        self.__x=x
        self.__y=y
        
    def r2_score(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name_r:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbr:
                print('CatBoost Regressor - ',r2_score(testy,pred))
                print()

            elif i==xgbr:
                print('XgBoost Regressor - ',r2_score(testy,pred))
                print()

            else:   
                print('{}      - '.format(i),r2_score(testy,pred))
                print()  
          
            
    
            
    def cross_validation(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name_r:
            i.fit(trainx,trainy)
            score=cross_val_score(i,self.__x,self.__y,cv=5)
            if i==cbr:
                print('CatBoost Regressor - ',score.mean())
                print()

            elif i==xgbr:
                print('XgBoost Regressor - ',score.mean())
                print()

            else:   
                print('{}     - '.format(i),score.mean())
                print()
                

    def mean_squared_error(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.20,random_state=42)
        for i in name_r:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbr:
                print('CatBoost Regressor - ',mean_squared_error(testy,pred))
                print()

            elif i==xgbr:
                print('XgBoost Regressor - ',mean_squared_error(testy,pred))
                print()

            else:   
                print('{}     - '.format(i),mean_squared_error(testy,pred))
                print()

            

            
    def mean_absolute_error(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name_r:
            i.fit(trainx,trainy)
            pred=i.predict(testx)
            if i==cbr:
                print('CatBoost Regressor - ',mean_absolute_error(testy,pred))
                print()

            elif i==xgbr:
                print('XgBoost Regressor - ',mean_absolute_error(testy,pred))
                print()

            else:   
                print('{}      - '.format(i),mean_absolute_error(testy,pred))
                print()

            
            
    def overfitting(self):
        trainx,testx,trainy,testy=train_test_split(self.__x,self.__y,test_size=0.25,random_state=42)
        for i in name_r:
            i.fit(trainx,trainy)
            pred_train=i.predict(trainx)
            pred_test=i.predict(testx)
            if i==cbr:
                print('Training Accuracy of CatBoost Regressor    - ',r2_score(testy,pred_test))
                print('Testing Accuracy  of CatBoost Regressor    - ',r2_score(testy,pred_test))
                print()

            elif i==xgbr:
                print('Training Accuracy of XgBoost Regressor    - ',r2_score(testy,pred_test))
                print('Testing Accuracy  of XgBoost Regressor    - ',r2_score(testy,pred_test))
                print()

            else:

                print('Training Accuracy of {}     - '.format(i),r2_score(trainy,pred_train))
                print('Testing Accuracy  of {}     - '.format(i),r2_score(testy,pred_test))
                print()