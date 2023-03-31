Pratik model

- The best thing about this package is that you don’t have to train and predict every 
classification or regression algorithm to check performance. 
- This package directly gives you output performance on 13 different algorithms.

For Classification
from Pratik_model import smart_classifier

model = smart_classifier(x,y)
model.accuracy_score()
model.classification_report()
model.confusion_matrix()
model.cross_validation()
model.mean_absolute_error()
model.precision_score()
model.recall_score()
model.mean_absolute_error()
model.mean_absolute_error()
model.mean_squared_error()
model.cross_validation()

For Regression 
from Pratik_model import smart_regressor

model=smart_regressor(x,y)
model.r2_score()
model.mean_absolute_error()
model.mean_absolute_error()
model.mean_squared_error()
model.cross_validation()
model.overfitting()


For more details check Source code .

First Release
0.0.2 (29/3/2022)

Thank You!!.