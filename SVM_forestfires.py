
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\SVM\\forestfires.csv")

df=df.iloc[:,[30,6,7,8,9]]

df['size_category'].value_counts()

X=df.iloc[:,1:5]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.preprocessing import scale

X_train=scale(X_train)
X_test=scale(X_test)


#Model building step:

model=SVC(kernel='linear')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
###Matplot Lib for 1st model
import matplotlib.pylab as plt
plt.scatter(x=X,y= y,color= 'red');plt.plot(X,y_pred,color ='black')
py.plot(X,y_pred,color = 'black')
plt.title('1st Model')



py.hist(x)
py.boxplot(x,0,"rs",0)
py.hist(y_pred)
py.boxplot(y_pred,0,"rs",0)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

#######################################################################################################

model2=SVC(kernel='rbf')
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)

plt.scatter(x=X,y= y,color= 'red');plt.plot(X,y_pred2,color ='black')
py.plot(X,y_pred2,color = 'black')
plt.title('2nd Model')



py.hist(x)
py.boxplot(x,0,"rs",0)
py.hist(y_pred2)
py.boxplot(y_pred2,0,"rs",0)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred2)
confusion_matrix(y_test,y_pred2)
classification_report(y_test,y_pred2)

#########################################################################################################

model2=SVC(kernel='poly')
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy_score(y_test,y_pred2)
confusion_matrix(y_test,y_pred2)
classification_report(y_test,y_pred2)

