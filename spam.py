from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy as sp
f = open('spambase/spambase.data', 'r')
strings = f.read().split('\n')[0:-1]
f.close()
np.random.shuffle(strings)
rows = len(strings)
columns = len(strings[0].split(',')) - 1
set = sp.zeros((rows, columns))
answers = sp.zeros((rows))
for row in range(rows):
	set[row] = strings[row].split(',')[0:-1]
	answers[row] = strings[row].split(',')[-1]
feature_means = sp.mean(set, axis=0) 
set -= feature_means 
feature_std = sp.std(set, axis=0) 
set /= feature_std
train_rows = rows // 2
train_set = sp.zeros((train_rows, columns))
train_answers = sp.zeros((rows))
test_set = sp.zeros((rows - train_rows, columns))
test_answers = sp.zeros((rows - train_rows))
train_set = set[:train_rows]
test_set = set[train_rows:rows]
train_answers = answers[:train_rows]
test_answers = answers[train_rows:rows]
new_classifier = MLPClassifier().fit(X = train_set, y = train_answers);
#train_errors = np.sum(new_classifier.predict(train_set) != train_answers)
#print ('%f percent errors in the train set. (Using your own train set fot testing is cheating, but you probably dont care anyway)' % (train_errors * 100.0 / len(test_set)))
test_errors = np.sum(new_classifier.predict(test_set) != test_answers)
print ('%f percent errors in the test set' % (test_errors * 100.0 / len(test_set)))
TP = sp.sum(new_classifier.predict(test_set) * test_answers)
TN = sp.sum((1 - new_classifier.predict(test_set)) * (1 - test_answers))
FP = sp.sum(new_classifier.predict(test_set) * (1 - test_answers))
FN = sp.sum((1 - new_classifier.predict(test_set)) * test_answers)
Sensitivity = TP / (TP + FN)
Specitivity = TN / (TN + FP)
Precision = TP / (TP + FP)
F1 = 2 * Precision * Sensitivity / (Precision + Sensitivity)
print ('%d true positives, %d true negatives' % (TP, TN))
print ('%d false positives, %d false negatives' % (FP, FN))
print ('Sensitivity: %f, Specitivity: %f' % (Sensitivity, Specitivity))
print ('Precision: %f, Recall: %f' % (Precision, Sensitivity))
print ('F1: %f' % F1)
