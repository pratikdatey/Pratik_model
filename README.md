Pratik_model
- The best thing about this package is that you do not have to train and predict every classification or regression algorithm to check performance.
- This package directly gives you output performance on 13 different algorithms.

How to use it  - 
For Classification
x= Independent variables
y= Dependent variables

* From Pratik_model import smart_classifier
* model = smart_classifier(x,y)
* model.accuracy_score()
* model.classification_report()
* model.confusion_matrix()
* model.cross_validation()
* model.mean_absolute_error()
* model.precision_score()
* model.recall_score()
* model.mean_absolute_error()
* model.mean_absolute_error()
* model.mean_squared_error()
* model.cross_validation()

For Regression -

* From Pratik_model import smart_regressor
* model=smart_regressor(x,y)
* model.r2_score()
* model.mean_absolute_error()
* model.mean_absolute_error()
* model.mean_squared_error()
* model.cross_validation()
* model.overfitting()

Check Pratik_Model_Package.ipynb file on Github for practical code.

Pratik_model for Classification: 
It will check the performance on this Classification models:
- Passive Aggressive Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Logistic Regression
- Ridge Classifier
- K Neighbors Classifier
- Support Vector Classification
- Naive Bayes Classifier
- LGBM Classifier
- CatBoost Classifier
- XGB Classifier

And for classification problems Pratik_model can give the output of:
- Accuracy Score.
- Classification Report
- Confusion Matrix
- Cross validation (Cross validation score)
- Mean Absolute Error
- Mean Squared Error
- Overfitting (will give accuracy of training and testing data.)
- Precision Score
- Recall Score

Pratik_model for Regression: 
Similarly, It will check performance on this Regression model:
- Passive Aggressive Regressor
- Gradient Boosting Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Lasso Regression
- K Neighbors Regressor
- Linear Regression
- Support Vector Regression
- LGBM Regressor
- CatBoost Regressor
- XGB Regressor

And for Regression problem Pratik_model
can give an output of:
- R2 Score.
- Cross validation (Cross validation score)
- Mean Absolute Error
- Mean Squared Error
- Overfitting (will give accuracy of training and testing data.)


First Release
0.0.7 (29/3/2022)

Thank You!!.
