import arff
import codecs
import weka.core.jvm as jvm 
jvm.start()
from sklearn.tree import DecisionTreeClassifier
from weka.classifiers import Evaluation, Classifier
from weka.core.converters import Loader
import sys
import numpy as np
import pickle

file=""
attributes=[]
dt = DecisionTreeClassifier()
cls = Classifier(classname="weka.classifiers.trees.J48")
def case1() :
    global file
    global dt
    global cls
    loader = Loader(classname="weka.core.converters.ArffLoader")
    file = input("Enter the name of the file:")
    data = loader.load_file(file)
    data.class_is_last()
    cls.build_classifier(data)
    file_ = codecs.open(file, 'rb', 'utf-8')
    data = arff.load(file_)
    data2,data3 = [],[]
    target = []
    data1 = data['data']
    for i in data['attributes']:
        attributes.append(i[0])
    attributes.pop()	
    for row in data1:
        target.append(row.pop())
        data2.append(row)
    dt.fit(data2,target)
    pickle.dump(dt,open(input("please enter model file name: ")+".sav",'wb'))
    print("Model created")

def case2() :
    loader1 = Loader(classname="weka.core.converters.ArffLoader")
    test_file = input("Enter the name of the test file:")
    data1=loader1.load_file(test_file)
    data1.class_is_last()
    evaluation = Evaluation(data1)
    evl = evaluation.test_model(cls, data1)
    print(evaluation.matrix("=== (confusion matrix) ==="))

def case3() :
    global attributes
    attributes=[]
    file_ = codecs.open(file, 'rb', 'utf-8')
    data = arff.load(file_)
    data3 = []
    for i in data['attributes']:
        attributes.append(i[0])
    attributes.pop()	
    while(1):
        ch = int(input("1.Enter values interactively 2.Quit : "))
        if ch == 1 :
            for i in attributes:
                data3.append(input("Please enter a value for "+i+" ")) 
            predict=dt.predict(np.reshape(data3,(1,-1)))
            print(predict[0])
            data3=[]			
        else :
           break


def case4() : 
    loader1 = Loader(classname="weka.core.converters.ArffLoader")
    file = input("Enter the name of the  model file:")
    dt = pickle.load(open("iris.arff.sav", 'rb'))
    data3 = []
    while(1):
        ch = int(input("1.Enter values interactively 2.Quit : "))
        if ch == 1 :
            for i in attributes:
                data3.append(input("Please enter a value for "+i+" "))				
            predict=dt.predict(np.reshape(data3,(1,-1)))
            print(predict[0])
            data3=[]			
        else :
           break
		   
while(1):
    print("========================================================================================================")
    print("1. Learn a Decision Tree and Save the tree \n2. Testing the accuracy of the decision tree \n3. Applying the decision tree to new cases \n4. Load a new Model \n5. Quit.")
    print("========================================================================================================")
    choice = int(input("Enter your choice:"))
    while choice not in [1,2,3,4,5]:
        choice = int(input("Invalid choice.\n 1. Learn a Decision Tree and Save the tree \n2. Testing the accuracy of the decision tree \n3. Applying the decision tree to new cases \n4. Load a new Model \n5. Quit."))
    if choice == 1 :
        case1()
    elif choice == 2 :
        case2()
    elif choice == 3 :
        case3()
    elif choice == 4 :
        case4()
    else :
        print("The program will now terminate!")
        jvm.stop()		
        sys.exit()