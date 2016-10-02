from logreg import LogisticRegression
from random import shuffle
import numpy as np
f = open('spambase/spambase.data', 'r')
strings = f.read().split('\n')[0:-1]
f.close()
shuffle(strings)
train_rows = len(strings) // 2
columns = len(strings[0].split(',')) - 1
train_set = np.zeros((train_rows, columns))
train_answers = np.zeros((train_rows))
test_rows = len(strings) - train_rows
test_set = np.zeros((test_rows, columns))
test_answers = np.zeros(test_rows)
for row in range(test_rows - 1):
	train_set[row] = strings[row].split(',')[0:-1]
	train_answers[row] = strings[row].split(',')[-1]
for row in range(test_rows, len(strings)):
	test_set[row - train_rows] = strings[row].split(',')[0:-1]
	test_answers[row - train_rows] = strings[row].split(',')[-1]
feature_means = np.mean(train_set, axis=0) 
train_set -= feature_means 
feature_std = np.std(train_set, axis=0) 
train_set /= feature_std
new_classifier = LogisticRegression()
new_classifier.fit(X = train_set, y = train_answers, max_iters = 1000)
train_errors = np.sum(new_classifier.predict(train_set) != train_answers)
print ('%f percent errors in the train set. (Using your own train set fot testing is cheating, but you probably dont care anyway)' % (train_errors * 100.0 / len(test_set)))
test_errors = np.sum(new_classifier.predict(test_set) != test_answers)
print ('%f percent errors in the test set' % (test_errors * 100.0 / len(test_set)))