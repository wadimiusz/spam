from logreg import LogisticRegression
from random import shuffle
import numpy as np
f = open('spambase/spambase.data', 'r')
strings = f.read().split('\n')[0:-1]
f.close()
shuffle(strings)
rows = len(strings)
columns = len(strings[0].split(',')) - 1
set = np.zeros((rows, columns))
answers = np.zeros((rows))
for row in range(rows):
	set[row] = strings[row].split(',')[0:-1]
	answers[row] = strings[row].split(',')[-1]
feature_means = np.mean(set, axis=0) 
set -= feature_means 
feature_std = np.std(set, axis=0) 
set /= feature_std
train_rows = rows // 2
train_set = np.zeros((train_rows, columns))
train_answers = np.zeros((rows))
test_set = np.zeros((rows - train_rows, columns))
test_answers = np.zeros((rows - train_rows))
train_set = set[1:train_rows]
test_set = set[train_rows:rows]
train_answers = answers[1:train_rows]
test_answers = answers[train_rows:rows]
new_classifier = LogisticRegression()
new_classifier.fit(X = train_set, y = train_answers, max_iters = 1000)
#train_errors = np.sum(new_classifier.predict(train_set) != train_answers)
#print ('%f percent errors in the train set. (Using your own train set fot testing is cheating, but you probably dont care anyway)' % (train_errors * 100.0 / len(test_set)))
test_errors = np.sum(new_classifier.predict(test_set) != test_answers)
print ('%f percent errors in the test set' % (test_errors * 100.0 / len(test_set)))