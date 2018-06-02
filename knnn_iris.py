from  sklearn.datasets  import load_iris
from  sklearn.model_selection  import  train_test_split
from sklearn.neighbors  import  KNeighborsClassifier

#  loading  data 
iris=load_iris()

train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)

#  calling KNN  classifier 
knn=KNeighborsClassifier(n_neighbors=5)

#  loading  data in KNN
trained=knn.fit(train_data,train_target)

#  predicting model 
predicted=trained.predict(test_data)

#print(predicted)

#  accuracy  test
from  sklearn.metrics import  accuracy_score
rate=accuracy_score(test_target,predicted)
print(rate)
