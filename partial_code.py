from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def my_tokenizer(s):
    return s.split()
    

clf_nb = MultinomialNB()
clf_svm = svm.LinearSVC(C=10)
clf_log = LogisticRegression(C=100, penalty='l2', solver='liblinear')
clf_rdf = RandomForestClassifier(n_jobs=-1, n_estimators=10)

#clf = clf_log


skf = StratifiedKFold(n_splits=10, random_state=0)
accuracies = []
precisions = []
recalls = []
f1s = []
kappas = []
i = 0
for train_index, test_index in skf.split(corpus, labels):
    print('Fold :',i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(data_train)
    cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
#    cs = [5,10,15,20] #Random forrest
   # cs = [1, 2, 3, 5, 10] #KNN
    best_c = 0
    best_score = 0
    for c in cs:
        clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
        #clf_inner = svm.LinearSVC(C=c)
        #clf_inner = RandomForestClassifier(n_estimators=c, n_jobs=-1)
        #clf_inner = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
        sub_skf = StratifiedKFold(n_splits=3, random_state=0)
        scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
        score = np.mean(scores_inner)
        if score > best_score:
            best_score = score
            best_c = c
    clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=best_c)
    #clf = MultinomialNB()
    #clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    #clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    clf.fit(train_tfidf, labels_train)
    test_tfidf = vec.transform(data_test)
    predicted = clf.predict(test_tfidf)
    accuracy = metrics.accuracy_score(labels_test, predicted)
    precision = metrics.precision_score(labels_test, predicted, average='macro')
    recall = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kappa = metrics.cohen_kappa_score(labels_test, predicted)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1_macro)
    kappas.append(kappa)
    i += 1

    
#skf = StratifiedKFold(n_splits=10, random_state=3)
#scores_txt = cross_val_score(clf, corpus_tfidf, labels, scoring='f1_macro', cv=skf)
print(prob[0].upper()+prob[1:]+' Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies), np.std(accuracies) * 2))
print(prob[0].upper()+prob[1:]+' Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions), np.std(precisions) * 2))
print(prob[0].upper()+prob[1:]+' Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls), np.std(recalls) * 2))
print(prob[0].upper()+prob[1:]+' F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print(prob[0].upper()+prob[1:]+' Kappa: %0.2f (+/- %0.2f)' % (np.mean(kappas), np.std(kappas) * 2))