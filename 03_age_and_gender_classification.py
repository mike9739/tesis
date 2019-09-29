#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from scipy.sparse import hstack
import numpy as np
import time
#import warnings



def classification_report_with_f1_score(y_true, y_pred):
    print(metrics.confusion_matrix(y_true, y_pred))
    #print(classification_report(y_true, y_pred)) # print classification report
    return f1_score(y_true, y_pred, average='macro') # return accuracy score

def my_tokenizer(s):
    return s.split()

def read_users(file):
    users = []
    with open(file) as content_file:
        for line in content_file:
            users.append(line.strip())
    return users

def read_labels(file, labels_names):
    label = []
    with open(file) as content_file:
        for line in content_file:
            category = line.strip()
            if(category == 'secundaria' or category == 'preparatoria'):
                category='media'
            else:
                category = 'superior'
            
            label.append(labels_names.index(category))
    return label

def clean_words(words, stop_words):
    text = ' '.join([word for word in words if len(word)>2 and len(word)<35 and word not in stop_words])
    return text

def read_text_data_with_emos(text_file, emo_file):
    data = []
   
    stop_words = stopwords.words('spanish')
   
    with open(text_file) as text_content, open(emo_file) as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            text += ' '+emo_line.rstrip()
            data.append(text)
    return data

def read_emos(emo_file):
    data = []
    with open(emo_file) as emo_content:
        for emo_line in emo_content:
            emo = emo_line.rstrip()
            data.append(emo)
    return data

def read_text_data(file):
    data = []
    
    stop_words = stopwords.words('spanish')
   
    with open(file) as content_file:
        for line in content_file:
            words = line.rstrip().split()
            text = clean_words(words, stop_words)
            data.append(text)
    return data

def read_extra_data(n, file):
    data = []
    with open(file) as content_file:
        for line in content_file:
            tokens = line.rstrip().split()
            #ratio = len(tokens)/n
            ratio = len(tokens)
            data.append(ratio)
    return data

def group_per_user(corpus, labels, users):
    corpus_grouped = []
    labels_grouped = []
    d_text = {}
    d_label = {}
    for user,label,text in zip(users, labels, corpus):
        d_text[user] = d_text.get(user,'')+text+' '
        d_label[user] = label
    for user in d_text:
        corpus_grouped.append(d_text[user])
        labels_grouped.append(d_label[user])
    return corpus_grouped, labels_grouped

def group_extra_per_user(data, labels, users):
    data_grouped = [0]*(max(users)+1)
    labels_grouped = [0]*(max(users)+1)
    for user, label, datum in zip(users, labels, data):
        data_grouped[user] += datum
        labels_grouped[user] = label
    return data_grouped, labels_grouped

def roc_auc_multiclass(label_test, predicted):
    lb = LabelBinarizer()
    lb.fit(label_test)
    y_test = lb.transform(label_test)
    y_pred = lb.transform(predicted)
    return metrics.roc_auc_score(y_test, y_pred, average='macro')
    

lang = 'spa'
year = 2015
main_dir = '/home/miguel/Documentos/tesis/'
prob = ''  
labels_file = main_dir+'nivel_educativo.txt'
words_file = main_dir+'DataSetTest_words.txt'
tweets_file = main_dir+'tweets.txt'
users_file = main_dir+'names.txt'
hashs_file = main_dir+'DataSetTest_hashtags.txt'
ats_file = main_dir+'DataSetTest_ats.txt'
emo_file = main_dir+'DataSetTest_emoticons.txt'
links_file = main_dir+'DataSetTest_links.txt'
labels_names = ['media', 'superior']
 
 


labels_list = read_labels(labels_file, labels_names)
users_list = read_users(users_file)
corpus = []
corpus = read_text_data( words_file)
#corpus = read_text_data_with_emos(lang, words_file, emo_file)
#corpus = read_emos(emo_file)
corpus, labels_list = group_per_user(corpus, labels_list, users_list)
labels = np.asarray(labels_list)
labels_set = set(labels_list)

#vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
#corpus_tfidf = vec.fit_transform(corpus)


#clf_nb = MultinomialNB()
#clf_svm = svm.LinearSVC(C=10)
#clf_log = LogisticRegression(C=100, penalty='l2', solver='liblinear')
#clf_rdf = RandomForestClassifier()
#clf_knn = KNeighborsClassifier()

#clf = clf_log


skf = StratifiedKFold(n_splits=10, random_state=0) #10 
scores_accuracy = []
scores_precission_macro = []
scores_recall_macro = []
scores_f1_macro = []
scores_kapha = [] 
scores_roc = []


i = 0

start = time.time()

for train_index, test_index in skf.split(corpus, labels):
    print('Fold :',i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(data_train)
    
    
#    ks = [1, 2, 3, 5, 10]
#    cs = [0.01, 0.1, 1, 10, 100]
#    ts = [5,10,15,20]
#    best_c = 0
#    best_score = 0
#    best_k = 0
#    best_t=0   
#    
#
#    for k in ks:
#        #print(c)
#        #warnings.filterwarnings('ignore')
#        clf_inner = KNeighborsClassifier(n_neighbors=k)
#        sub_skf = StratifiedKFold(n_splits=3, random_state=0)
#        scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
#        #scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring=make_scorer(classification_report_with_f1_score), cv=sub_skf)
#        score = np.mean(scores_inner)
#        #print(score)
#        if score > best_score:
#            best_score = score
#            best_k = k
            
    #clf =LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = KNeighborsClassifier(n_neighbors=best_k)
    clf = MultinomialNB()
   # clf = RandomForestClassifier(n_estimators=best_t)
   #clf = svm.LinearSVC(C=best_c)
    clf.fit(train_tfidf, labels_train)
    test_tfidf = vec.transform(data_test)
    predicted = clf.predict(test_tfidf)
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted)
    #roc = metrics.roc_curve(labels_test, predicted)
    roc = metrics.roc_auc_score(labels_test,predicted)
    #roc = roc_auc_multiclass(labels_test, predicted)
    
    print(metrics.confusion_matrix(labels_test, predicted))
    
    #scores.append(f1_macro)
    scores_accuracy.append(accuracy)
    scores_precission_macro.append(precission_macro)
    scores_recall_macro.append(recall_macro)
    scores_f1_macro.append(f1_macro)
    scores_kapha.append(kapha)
    scores_roc.append(roc)
    i += 1

end = time.time()

    
#skf = StratifiedKFold(n_splits=10, random_state=3)
#scores_txt = cross_val_score(clf, corpus_tfidf, labels, scoring='f1_macro', cv=skf)
print(' Accuracy: %0.2f (+/- %0.2f)' % (np.mean(scores_accuracy), np.std(scores_accuracy) * 2))
print(' Precssion: %0.2f (+/- %0.2f)' % (np.mean(scores_precission_macro), np.std(scores_precission_macro) * 2))
print(' Recall: %0.2f (+/- %0.2f)' % (np.mean(scores_recall_macro), np.std(scores_recall_macro) * 2))
print(' F1: %0.2f (+/- %0.2f)' % (np.mean(scores_f1_macro), np.std(scores_f1_macro) * 2))
print(' Kapha: %0.2f (+/- %0.2f)' % (np.mean(scores_kapha), np.std(scores_kapha) * 2))
print(' ROC: %0.2f (+/- %0.2f)' % (np.mean(scores_roc), np.std(scores_roc) * 2))
print("Time of training + testing: %0.2f " % (end - start))

#n_text = len(corpus)
#labels_list = read_labels(labels_file, labels_names)
#hashs = read_extra_data(n_text, hashs_file)
#ats = read_extra_data(n_text, ats_file)
#emoticons = read_extra_data(n_text, emo_file)
#links = read_extra_data(n_text, links_file)
#
#hashs, labels = group_extra_per_user(hashs, labels_list, users_list)
#ats = group_extra_per_user(ats, labels_list, users_list)[0]
#emoticons = group_extra_per_user(emoticons, labels_list, users_list)[0]
#links = group_extra_per_user(links, labels_list, users_list)[0]
#
#feat = [(a,b,c,d) for (a,b,c,d) in zip(hashs,ats,emoticons,links)]
#feat = [list(a) for a in feat]
#
#feat = np.array(feat, dtype='float')
#feat = normalize(feat, norm='l2', axis=1)
#scores = cross_val_score(clf, feat, labels, cv=skf)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#feat_tot = hstack([corpus_tfidf, feat])
#feat_tot = normalize(feat_tot, norm='l2', axis=1)
#scores = cross_val_score(clf_nb, feat_tot, labels, cv=skf)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#voc_file = main_dir+'vocabulary.txt'
#with open(voc_file,'w') as voc_writer:
#    for word in vec.vocabulary_:
#        voc_writer.write(word+'\n')


#for train_index, test_index in skf.split(corpus_tfidf, labels):
#    data_train, data_test = corpus_tfidf[train_index], corpus_tfidf[test_index]
#    labels_train, labels_test = labels[train_index], labels[test_index]
#    break