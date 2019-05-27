from mnist import MNIST
import numpy as np
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

data = MNIST('samples')

training_size = 3000
testing_size = 1000

train_images, train_labels = data.load_training()
test_images, test_labels = data.load_testing()

train_images = np.array(train_images[:training_size])
train_labels = np.array(train_labels[:training_size])

test_images = np.array(test_images[:testing_size])
test_labels = np.array(test_labels[:testing_size])

#Rescaling Data"""￼

train_images = train_images/255
test_images = test_images/255

clf=DecisionTreeClassifier()
clf.fit(train_images,train_labels)

p=clf.predict(test_images)

count =0
for i in range (0,1000):
    if p[i]==test_labels[i]:
        count+=1
print("Accuracy=", (count/1000)*100)


#♣---------------------------

rf=RandomForestClassifier(n_estimators=100)
rf.fit(train_images,train_labels)
pred=rf.predict(test_images)

countt =0
for i in range (0,1000):
    if pred[i]==test_labels[i]:
        countt+=1
print("Accuracy=", (countt/1000)*100)



def demo(j): #44,8,102
    print("Digit is ",test_labels[j])
    a1=test_images[j]
    a1=a1.reshape(28,28)
    pt.imshow(255-a1,cmap='gray')
    pt.savefig("tests.png", dpi=300)
    pt.show()
    print("Decision tree prediction is ",clf.predict([test_images[j]]))
    print("Randomforest prediction is ",rf.predict([test_images[j]]))

demo(2)
