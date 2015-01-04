#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import ensemble
from sklearn import datasets
from sklearn import linear_model
from sklearn import naive_bayes 
from sklearn import dummy 
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from operator import itemgetter, attrgetter, methodcaller
from datetime import date, datetime

from sklearn.grid_search import GridSearchCV


try:
	import ujson as json
except ImportError:
	try:
		import simplejson as json
	except ImportError:
		import json

###############################################################################
# Load data
with open("features.json") as f:
	commutes = json.load(f)


def format_data(commutes):
	data = []
	labels = []
	features = []

	for c in commutes:
		labels.append(c.pop("commute_duration_seconds"))
		dc = []
		features = []
		for feature in sorted(c):
			features.append(feature)
			dc.append(c[feature])
		data.append(dc)

	return np.array(data), np.array(labels), np.array(features)

commutes_data, commutes_labels, commutes_feature_labels = format_data(commutes)

X, y = shuffle(commutes_data, commutes_labels, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

###############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}

classifiers = {}
classifiers['GradientBoostingRegressor'] = ensemble.GradientBoostingRegressor(**params)
classifiers['Dummy'] = dummy.DummyClassifier(strategy="stratified")
classifiers['LinearRegression'] = linear_model.LinearRegression()
classifiers['Ridge'] = linear_model.Ridge (alpha = .5)
classifiers['LogisticRegression'] = linear_model.LogisticRegression(C=1, penalty='l2', tol=0.01)
classifiers['SGDClassifier'] = linear_model.SGDClassifier(loss="hinge", penalty="l2")
classifiers['AdaBoost'] = ensemble.AdaBoostClassifier(n_estimators=100)
classifiers['RandomForestClassifier'] = ensemble.RandomForestClassifier(n_estimators=10)
#classifiers['NearestNeighbors'] = neighbors.NearestNeighbors(n_neighbors=2, algorithm='ball_tree')                       
classifiers['GaussianNB'] = naive_bayes.GaussianNB()

classifier_results = {}
for name in classifiers:
	scores = cross_validation.cross_val_score(classifiers[name], X, y)
	classifier_results[name] = scores
	classifiers[name].fit(X_train, y_train)

for name, scores in sorted(classifier_results.items(), key=lambda x: x[1].mean(), reverse=True):
	print("Accuracy(%s): %0.2f (+/- %0.2f)" % (name, scores.mean(), scores.std() * 2))

#[u'day_of_month' u'day_of_week' u'direction' u'month_of_year' u'starting_time_hour' u'starting_time_minute' u'year']
dt_now = datetime.now()
now = np.array([[dt_now.month, dt_now.weekday(), 0, dt_now.month, dt_now.hour, dt_now.minute, dt_now.year],
	    		[dt_now.month, dt_now.weekday(), 1, dt_now.month, dt_now.hour, dt_now.minute, dt_now.year]])
now_classifications = classifiers["GradientBoostingRegressor"].predict(now)
print "current commute time to work is predected to be {min} minutes".format(min=now_classifications[0]/60)
print "current commute time to home is predected to be {min} minutes".format(min=now_classifications[1]/60)


###############################################################################
# Plot training deviance

clf = classifiers["GradientBoostingRegressor"]
print "writing model to disk"
from sklearn.externals import joblib
joblib.dump(clf, 'commute_model.pkl') 


# # compute test set deviance
# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

# for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')

# ###############################################################################
# # Plot feature importance
# feature_importance = clf.feature_importances_
# # make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, commutes_feature_labels[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
# plt.show()