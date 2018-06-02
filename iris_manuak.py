#!/usr/bin/python3
import time
#import matplotlib.pyplot as plt
from sklearn.datasets  import load_iris
from  sklearn  import  tree
import numpy  as np
#  loading  iris data
iris=load_iris()


x=[0,50,100]
#  removing  exactly one  data from each flower 
train_data=np.delete(iris.data,x,axis=0)

#  removing the same from target 
train_target=np.delete(iris.target,x)
#  testing  data  
test_data=iris.data[x]
test_target=iris.target[x]

#  calling  classifier 
clf=tree.DecisionTreeClassifier()
trained=clf.fit(train_data,train_target)

#  making  prediction 
predicted=trained.predict([[7. , 3.2, 4.7, 1.4]])
print(predicted)



'''
import  graphviz
out_data=tree.export_graphviz(clf,
                    out_file=None,
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    filled=True,
                    rounded=True
                    )
graphviz.Source(out_data)

'''
