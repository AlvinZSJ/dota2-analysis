import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import catboost as cb
import sklearn
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
'''
Default parameters lead good accuracy...

'''
train_dir = 'C:/Users/dell/Desktop/5012/nTrain.csv'
test_dir = 'C:/Users/dell/Desktop/5012/nTest.csv'
train = pd.read_csv(train_dir)
target = 'winner'
final_test = pd.read_csv(test_dir)
x_columns = [x for x in train.columns if x not in ['winner', 'cluster_id', 'game_mode', 'game_type']]
X = train[x_columns]
y = train['winner']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# model = cb.CatBoostClassifier(verbose=False, border_count=100,loss_function='Logloss',depth=10,learning_rate=0.15,l2_leaf_reg=1,iterations=1000)
model=cb.CatBoostClassifier(verbose=False)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
pred_score = accuracy_score(y_pred, y_test)
print('accuracy')
print(pred_score)
# 0.35

def acc(model):
    y_pred = model.predict(x_test)
    pred_score = accuracy_score(y_pred, y_test)
    print('accuracy')
    print(pred_score)


### KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
acc(clf)
# 0.5314799672935405 也很慢很慢

### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2')
clf.fit(x_train, y_train)
acc(clf)
# 0.5955846279640229

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
acc(clf)
# 0.5354701553556828


### Decision Tree Classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
acc(clf)
# 0.5149959116925593

### GBDT(Gradient Boosting Decision Tree) Classifier
from sklearn.ensemble import GradientBoostingClassifier
#
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(x_train, y_train)
acc(clf)
# 0.5894685200327064


# ### SVM Classifier
# from sklearn.svm import SVC
#
# clf = SVC(kernel='rbf', probability=True)
# clf.fit(x_train, y_train)
# acc(clf)
# 特别慢。。放弃了。。
