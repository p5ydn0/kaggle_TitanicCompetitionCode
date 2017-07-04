# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as preprocessing
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve

# 导入训练集数据
data_train = pd.read_csv('train.csv')

print '##################### Start!!!'
# 将缺失的Embarked取值为中位数(通过绘制图形后得到)
data_train.Embarked[data_train.Embarked.isnull()] = 'C'

# 处理Cabin特征变量(yes,no)
data_train.Cabin[data_train.Cabin.isnull()] = 'no'
data_train.Cabin[data_train.Cabin!='no'] = 'yes'

# 增加Famliy_size的属性
Famliy_size = data_train.SibSp + data_train.Parch + 1

# 增加Single属性
Single = Series(range(0,891),name='Single')
for i in range(0,len(Famliy_size)):
	if Famliy_size[i]==1:
		Single[i] = 1
	else:
		Single[i] = 0

# 增加Small_Family属性
Small_Family = Series(range(0,891),name='Small_Family')
for i in range(0,len(Famliy_size)):
	if (Famliy_size[i] > 1) and (Famliy_size[i] < 5):
		Small_Family[i] = 1
	else:
		Small_Family[i] = 0

# 增加Large_Family属性
Large_Family = Series(range(0,891),name='Large_Family')
for i in range(0,len(Famliy_size)):
	if  Famliy_size[i] > 4:
		Large_Family[i] = 1
	else:
		Large_Family[i] = 0

# 增加Mother属性
Mother = Series(range(0,891),name='Mother')
for i in range(0,891):
	if (data_train.Sex[i] == 'female') and (data_train.Parch[i] > 0) and ('Miss' not in data_train.Name[i]):
		Mother[i] = 1
	else:
		Mother[i] = 0

# Ticket属性处理
Ticket_class_list = []
for i in range(0,len(data_train.Ticket)):
	data_train.Ticket[i] = data_train.Ticket[i].replace('.','')
	data_train.Ticket[i] = data_train.Ticket[i].replace('/','')
	data_train.Ticket[i] = data_train.Ticket[i].split()
	data_train.Ticket[i][0].strip()
	if data_train.Ticket[i][0].isdigit():
		Ticket_class_list.append(0)
	else:
		Ticket_class_list.append(1)
	
Ticket_class = pd.DataFrame({u'Ticket_class':Ticket_class_list})

# 因子化处理
dummies_Cabin = pd.get_dummies(data_train.Cabin,prefix='Cabin')
dummies_Sex = pd.get_dummies(data_train.Sex,prefix='Sex')
dummies_Embarked = pd.get_dummies(data_train.Embarked,prefix='Embarked')
dummies_Pclass = pd.get_dummies(data_train.Pclass,prefix='Pclass')
dummies_Ticket_class = pd.get_dummies(Ticket_class,prefix='Ticket_class')

# 对Name进行处理
namelist = []
for i in range(0,len(data_train)):
	if ('Capt' in data_train.Name[i]) or ('Col' in data_train.Name[i]) or ('Major' in data_train.Name[i]) or ('Dr' in data_train.Name[i]) or ('Rev' in data_train.Name[i]):
		namelist.append(0)
	elif ('Jonkheer' in data_train.Name[i]) or ('Don' in data_train.Name[i]) or ('Sir' in data_train.Name[i]) or ('the Countess' in data_train.Name[i]) or ('Dona' in data_train.Name[i]) or ('Lady' in data_train.Name[i]):
		namelist.append(1)
	elif ('Mme' in data_train.Name[i]) or ('Ms' in data_train.Name[i]) or ('Mrs' in data_train.Name[i]):
		namelist.append(2)
	elif ('Mlle' in data_train.Name[i]) or ('Miss' in data_train.Name[i]):
		namelist.append(3)
	elif ('Mr' in data_train.Name[i]):
		namelist.append(4)
	elif ('Master' in data_train.Name[i]):
		namelist.append(5)		
	else:
		print '######################'
		print '######################'
		print 'namelist error happen!'
		print '######################'
		print '######################'
		
title = pd.DataFrame({u'title':namelist})

# 增加Child属性
childlist = []
for i in range(0,len(data_train)):
	if data_train.Age[i] <= 18:
		childlist.append(1)
	else:
		childlist.append(0)
Child = pd.DataFrame({u'Child':childlist})
dummies_Child = pd.get_dummies(Child.Child,prefix='Child')

data_train = pd.concat([data_train,dummies_Cabin,dummies_Sex,dummies_Embarked,dummies_Pclass,title,Single,Small_Family,Large_Family,dummies_Child,Mother,dummies_Ticket_class],axis=1)

print '##################### Age predicting......'

# 预测缺失的Age
median_all = data_train[data_train.Age.notnull()]['Age'].median()

median01_01 = data_train[data_train.Sex=='female'][data_train.Pclass==1][data_train.title==0][data_train.Age.notnull()]['Age'].median()
median01_02 = data_train[data_train.Sex=='female'][data_train.Pclass==1][data_train.title==1][data_train.Age.notnull()]['Age'].median()
median01_03 = data_train[data_train.Sex=='female'][data_train.Pclass==1][data_train.title==2][data_train.Age.notnull()]['Age'].median()
median01_04 = data_train[data_train.Sex=='female'][data_train.Pclass==1][data_train.title==3][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'female' and data_train.Pclass[i] == 1:
		if data_train.title[i] == 0:
			data_train.Age[i] = int(median01_01)
		elif data_train.title[i] == 1:
			data_train.Age[i] = int(median01_02)
		elif data_train.title[i] == 2:
			data_train.Age[i] = int(median01_03)
		elif data_train.title[i] == 3:
			data_train.Age[i] = int(median01_04)
		else:
			data_train.Age[i] = int(median_all)

median02_01 = data_train[data_train.Sex=='female'][data_train.Pclass==2][data_train.title==2][data_train.Age.notnull()]['Age'].median()
median02_02 = data_train[data_train.Sex=='female'][data_train.Pclass==2][data_train.title==3][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'female' and data_train.Pclass[i] == 2:
		if data_train.title[i] == 2:
			data_train.Age[i] = int(median02_01)
		elif data_train.title[i] == 3:
			data_train.Age[i] = int(median02_02)
		else:
			data_train.Age[i] = int(median_all)
			
median03_01 = data_train[data_train.Sex=='female'][data_train.Pclass==3][data_train.title==2][data_train.Age.notnull()]['Age'].median()
median03_02 = data_train[data_train.Sex=='female'][data_train.Pclass==3][data_train.title==3][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'female' and data_train.Pclass[i] == 3:
		if data_train.title[i] == 2:
			data_train.Age[i] = int(median03_01)
		elif data_train.title[i] == 3:
			data_train.Age[i] = int(median03_02)
		else:
			data_train.Age[i] = int(median_all)
			
median04_01 = data_train[data_train.Sex=='male'][data_train.Pclass==1][data_train.title==0][data_train.Age.notnull()]['Age'].median()
median04_02 = data_train[data_train.Sex=='male'][data_train.Pclass==1][data_train.title==1][data_train.Age.notnull()]['Age'].median()
median04_03 = data_train[data_train.Sex=='male'][data_train.Pclass==1][data_train.title==4][data_train.Age.notnull()]['Age'].median()
median04_04 = data_train[data_train.Sex=='male'][data_train.Pclass==1][data_train.title==5][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'male' and data_train.Pclass[i] == 1:
		if data_train.title[i] == 0:
			data_train.Age[i] = int(median04_01)
		elif data_train.title[i] == 1:
			data_train.Age[i] = int(median04_02)
		elif data_train.title[i] == 4:
			data_train.Age[i] = int(median04_03)
		elif data_train.title[i] == 5:
			data_train.Age[i] = int(median04_04)
		else:
			data_train.Age[i] = int(median_all)
			
median05_01 = data_train[data_train.Sex=='male'][data_train.Pclass==2][data_train.title==0][data_train.Age.notnull()]['Age'].median()
median05_02 = data_train[data_train.Sex=='male'][data_train.Pclass==2][data_train.title==4][data_train.Age.notnull()]['Age'].median()
median05_03 = data_train[data_train.Sex=='male'][data_train.Pclass==2][data_train.title==5][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'male' and data_train.Pclass[i] == 2:
		if data_train.title[i] == 0:
			data_train.Age[i] = int(median05_01)
		elif data_train.title[i] == 4:
			data_train.Age[i] = int(median05_02)
		elif data_train.title[i] == 5:
			data_train.Age[i] = int(median05_03)
		else:
			data_train.Age[i] = int(median_all)
			
median06_01 = data_train[data_train.Sex=='male'][data_train.Pclass==3][data_train.title==4][data_train.Age.notnull()]['Age'].median()
median06_02 = data_train[data_train.Sex=='male'][data_train.Pclass==3][data_train.title==5][data_train.Age.notnull()]['Age'].median()

for i in range(0,len(data_train.Age)):
	if data_train.Sex[i] == 'male' and data_train.Pclass[i] == 3:
		if data_train.title[i] == 4:
			data_train.Age[i] = int(median06_01)
		elif data_train.title[i] == 5:
			data_train.Age[i] = int(median06_02)
		else:
			data_train.Age[i] = int(median_all)
			
data_train = data_train.drop(['PassengerId','Cabin','Sex','Embarked','Pclass','Ticket','Name','Parch'],axis=1)

# 归一化处理(对Age和Fare)
scaler_age = preprocessing.scale(data_train.Age)
data_train.Age = scaler_age
scaler_fare = preprocessing.scale(data_train.Fare)
data_train.Fare = scaler_fare

print '##################### Model building......'

# 建立模型
data_train_np = data_train.as_matrix()
X = data_train_np[:,1:]
y = data_train_np[:,0]
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 1000, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
model = RandomForestClassifier(**parameters)
model.fit(X, y)

print '##################### Model building finished.Start test data process.'


# 对测试数据进行处理
data_test = pd.read_csv('test.csv')

# 对Cabin进行处理
data_test.Cabin[data_test.Cabin.isnull()] = 'no'
data_test.Cabin[data_test.Cabin != 'no'] = 'yes'

# 增加Famliy_size_test的属性
Famliy_size_test = data_test.SibSp + data_test.Parch + 1

# 增加Single_test属性
Single_test = Series(range(0,418),name='Single_test')
for i in range(0,len(Famliy_size_test)):
	if Famliy_size_test[i]==1:
		Single_test[i] = 1
	else:
		Single_test[i] = 0

# 增加Small_Family_test属性
Small_Family_test = Series(range(0,418),name='Small_Family_test')
for i in range(0,len(Famliy_size_test)):
	if (Famliy_size_test[i] > 1) and (Famliy_size_test[i] < 5):
		Small_Family_test[i] = 1
	else:
		Small_Family_test[i] = 0

# 增加Large_Family_test属性
Large_Family_test = Series(range(0,418),name='Large_Family_test')
for i in range(0,len(Famliy_size_test)):
	if  Famliy_size_test[i] > 4:
		Large_Family_test[i] = 1
	else:
		Large_Family_test[i] = 0

# 增加Mother属性
Mother_test = Series(range(0,418),name='Mother_test')
for i in range(0,418):
	if (data_test.Sex[i] == 'female') and (data_test.Parch[i] > 0) and ('Miss' not in data_test.Name[i]):
		Mother_test[i] = 1
	else:
		Mother_test[i] = 0

# 将缺失的Fare取值为中位数(数据来源于网络)
data_test.Fare[data_test.Fare.isnull()] = 8.05

# Ticket属性处理
Ticket_class_list_test = []
for i in range(0,len(data_test.Ticket)):
	data_test.Ticket[i] = data_test.Ticket[i].replace('.','')
	data_test.Ticket[i] = data_test.Ticket[i].replace('/','')
	data_test.Ticket[i] = data_test.Ticket[i].split()
	data_test.Ticket[i][0].strip()
	if data_test.Ticket[i][0].isdigit():
		Ticket_class_list_test.append(0)
	else:
		Ticket_class_list_test.append(1)
	
Ticket_class_test = pd.DataFrame({u'Ticket_class_test':Ticket_class_list_test})

# 因子化处理
dummies_test_Cabin = pd.get_dummies(data_test.Cabin,prefix = 'Cabin')
dummies_test_Embarked = pd.get_dummies(data_test.Embarked,prefix = 'Embarked')
dummies_test_Sex = pd.get_dummies(data_test.Sex,prefix = 'Sex')
dummies_test_Pclass = pd.get_dummies(data_test.Pclass,prefix = 'Pclass')
dummies_Ticket_class = pd.get_dummies(Ticket_class_test,prefix='Ticket_class_test')

# 对Name进行处理
namelist_test = []
for i in range(0,len(data_test)):
	if ('Capt' in data_test.Name[i]) or ('Col' in data_test.Name[i]) or ('Major' in data_test.Name[i]) or ('Dr' in data_test.Name[i]) or ('Rev' in data_test.Name[i]):
		namelist_test.append(0)
	elif ('Jonkheer' in data_test.Name[i]) or ('Don' in data_test.Name[i]) or ('Sir' in data_test.Name[i]) or ('the Countess' in data_test.Name[i]) or ('Dona' in data_test.Name[i]) or ('Lady' in data_test.Name[i]):
		namelist_test.append(1)
	elif ('Mme' in data_test.Name[i]) or ('Ms' in data_test.Name[i]) or ('Mrs' in data_test.Name[i]):
		namelist_test.append(2)
	elif ('Mlle' in data_test.Name[i]) or ('Miss' in data_test.Name[i]):
		namelist_test.append(3)
	elif ('Mr' in data_test.Name[i]):
		namelist_test.append(4)
	elif ('Master' in data_test.Name[i]):
		namelist_test.append(5)		
	else:
		print '###########################'
		print '###########################'
		print 'namelist_test error happen!'
		print '###########################'
		print '###########################'
		
title_test = pd.DataFrame({u'title_test':namelist_test})

# 增加Child属性
childlist_test = []
for i in range(0,len(data_test)):
	if data_test.Age[i] <= 18:
		childlist_test.append(1)
	else:
		childlist_test.append(0)
childlist_test = pd.DataFrame({u'childlist_test':childlist_test})
dummies_Child_test = pd.get_dummies(childlist_test.childlist_test,prefix='Child_test')

data_test = pd.concat([data_test,dummies_test_Cabin,dummies_test_Sex,dummies_test_Embarked,dummies_test_Pclass,title_test,Single_test,Small_Family_test,Large_Family_test,dummies_Child_test,Mother_test,dummies_Ticket_class],axis=1)

print '##################### Age of testdata predicting......'

# 预测缺失的Age
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'female' and data_test.Pclass[i] == 1:
		if data_test.title_test[i] == 0:
			data_test.Age[i] = int(median01_01)
		elif data_test.title_test[i] == 1:
			data_test.Age[i] = int(median01_02)
		elif data_test.title_test[i] == 2:
			data_test.Age[i] = int(median01_03)
		elif data_test.title_test[i] == 3:
			data_test.Age[i] = int(median01_04)
		else:
			data_test.Age[i] = int(median_all)
						
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'female' and data_test.Pclass[i] == 2:
		if data_test.title_test[i] == 2:
			data_test.Age[i] = int(median02_01)
		elif data_test.title_test[i] == 3:
			data_test.Age[i] = int(median02_02)
		else:
			data_test.Age[i] = int(median_all)
			
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'female' and data_test.Pclass[i] == 3:
		if data_test.title_test[i] == 2:
			data_test.Age[i] = int(median03_01)
		elif data_test.title_test[i] == 3:
			data_test.Age[i] = int(median03_02)
		else:
			data_test.Age[i] = int(median_all)
			
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'male' and data_test.Pclass[i] == 1:
		if data_test.title_test[i] == 0:
			data_test.Age[i] = int(median04_01)
		elif data_test.title_test[i] == 1:
			data_test.Age[i] = int(median04_02)
		elif data_test.title_test[i] == 4:
			data_test.Age[i] = int(median04_03)
		elif data_test.title_test[i] == 5:
			data_test.Age[i] = int(median04_04)
		else:
			data_test.Age[i] = int(median_all)
			
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'male' and data_test.Pclass[i] == 2:
		if data_test.title_test[i] == 0:
			data_test.Age[i] = int(median05_01)
		elif data_test.title_test[i] == 4:
			data_test.Age[i] = int(median05_02)
		elif data_test.title_test[i] == 5:
			data_test.Age[i] = int(median05_03)
		else:
			data_test.Age[i] = int(median_all)
			
for i in range(0,len(data_test.Age)):
	if data_test.Sex[i] == 'male' and data_test.Pclass[i] == 3:
		if data_test.title_test[i] == 4:
			data_test.Age[i] = int(median06_01)
		elif data_test.title_test[i] == 5:
			data_test.Age[i] = int(median06_02)
		else:
			data_test.Age[i] = int(median_all)
			
data_test = data_test.drop(['PassengerId','Cabin','Sex','Embarked','Pclass','Ticket','Name','Parch'],axis=1)
print '##################### Age of test data prediction finished processing......'

# 归一化处理  preprocessing.scale
data_test.Age = preprocessing.scale(data_test.Age)
data_test.Fare = preprocessing.scale(data_test.Fare)

print 'Result predicting......'

# 预测结果
result = model.predict(data_test.as_matrix())
data_test_02 = pd.read_csv('test.csv')
result_df = pd.DataFrame({'PassengerId':data_test_02.PassengerId.as_matrix(),'Survived':result.astype(np.int32)})
result_df.to_csv('RFResult1500.csv',index = False)

# # 输出交叉验证的评估
# print cross_validation.cross_val_score(bagging_clf,X,y,cv=5)
# 
# # 绘制学习曲线
# train_sizes,train_scores,test_scores = learning_curve(bagging_clf,X,y,cv=None,n_jobs=1,train_sizes=np.linspace(.05,1.,20),verbose=0)
# 
# train_scores_mean = np.mean(train_scores,axis=1)
# test_scores_mean = np.mean(test_scores,axis=1)
# 
# plt.figure()
# plt.plot(train_sizes,train_scores_mean,label=u'score of train data')
# plt.plot(train_sizes,test_scores_mean,label=u'score of test data')
# plt.legend()
# plt.title(u'learning_curve')
# plt.xlabel(u'data sizes')
# plt.ylabel(u'score')
# plt.show()

# 完成
print 'ok,done!'
