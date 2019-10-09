# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:48:56 2019

@author: mmval
"""
import gensim 
import numpy as np
from nltk.corpus import stopwords

from sklearn.preprocessing import normalize 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import numpy as np
import time
import codecs

"""
h=[.1,-.02,.32,-.4]
h=np.array(h)
w=[-.5,.006,.7,-.08]
w=np.array(w)
r=np.zeros(4)
r=h+w
print(r)
min_max_scaler = preprocessing.MinMaxScaler()
X= min_max_scaler.fit_transform(h.reshape(1,-1))
y= min_max_scaler.fit_transform(h.reshape(1,-1))
print(X)
print(y)
a=' '.join(str(item) for item in X[0])
print(a)

"""
#normalizar al final de terner la matriz


# Load Google's pre-trained Word2Vec model.

#deprecated
#model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)

#FOR SPAMASSASSIN
"""
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/'
email_file=main_dir+'SA_words.txt'
label_file=main_dir+'full_labels.txt'
"""

#PARA ENRON
"""
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/'
email_file=main_dir+'Enron_words.txt'
label_file=main_dir+'full_labels.txt'
"""

#PARA TREC
"""
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/'
email_file=main_dir+'trec07p_words.txt'
label_file=main_dir+'labels_trec07p.txt'
"""

#PARA SPAMACHIVE

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamArchive/'
email_file=main_dir+'SpamArchive_words.txt'
label_file=main_dir+'full_labels.txt'



z=[]
corpus=[]
labels=[]
#l=np.zeros(300)

with open(email_file,'r', encoding='utf-8') as reader, open(label_file,'r',encoding='utf-8') as label_reader:
   StopWords = stopwords.words('english')
   for i,line in enumerate(reader):
       tokens=line.rstrip().split()
       le=0
       l=np.zeros(300)
       for token in tokens:
           if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
               le=le+1
               #print('word ',token)
               #print(model[token][0])
               l=l+np.array(model[token])
       #print(l[0]) 
      
       if le>0:
           l=l/le
           corpus.append(l)
       else:
           z.append(i)
           
   h=1       
   for j,line in enumerate(label_reader):
        if j not in z:
            labels.append(line.strip())
        print(h)
        h=h+1


labels=np.array(labels)  

corpus=np.array(corpus)

#FOR NB
for i in range(len(corpus)):
    corpus[i]=(corpus[i]-np.amin(corpus[i]))/(np.amax(corpus[i])-np.amin(corpus[i]))

corpus=normalize(corpus, norm='l2')



#archivo.write(str(corpus))

#h=1
#with open(label_file,'r',encoding='utf-8') as label_reader:
 #   for line in label_reader:
  #      labels.append(line.strip())
   #     print(h)
    #    h=h+1
        
#labels=np.array(labels)     

train = time.time()
clf_nb = MultinomialNB()
clf_svm = svm.LinearSVC(C=10)
clf_log = LogisticRegression(C=100, penalty='l2', solver='liblinear')
clf_rdf = RandomForestClassifier(n_jobs=-1, n_estimators=10)

#clf = clf_log

#indica en cuantas partes se dividiran los datos
skf = StratifiedKFold(n_splits=10, random_state=0)
accuracies = []
precisions = []
recalls = []
f1s = []
kappas = []
scores_roc=[]
i = 0

for train_index, test_index in skf.split(corpus, labels):
    print('Fold: ',i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    
    #min_dif=1 ignora palabras que aparezcan menos de una vez
    #norm=l1 norma utilizada para normalizar los vectores
    #analyzer=word indica que las caracteristicas estaran echas de palabras
    #tokenizer= my_tokenizer separa por palabras 
    #vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    
    #aprende el vocabulario y la frecuencia inversa del documento (idf), 
    #regresa la matriz termino-documento 
    #train_tfidf = vec.fit_transform(data_train)#
    
    """
    #cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
    #cs = [5,10,15,20] #Random forrest
    cs = [1, 2, 3, 5, 10] #KNN
    best_c = 0
    best_score = 0
    for c in cs:
        
        #
        #penalty: especifica la norma utilizada para la penalizacion 
        #solver: algoritmo usado para la optimizacion del problema
        #clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
       
        #clf_inner = svm.LinearSVC(C=c)
        
        #clf_inner = RandomForestClassifier(n_estimators=c, n_jobs=-1)
        
        clf_inner = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
        
        #clf_inner: objeto utilizado para ajustar los datos
        #los datos a ajustar
        #labels_train: variable objetivo
        #
        #cv:cross-validation splitting strategy
        sub_skf = StratifiedKFold(n_splits=3, random_state=0)
        scores_inner = cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=sub_skf)
        score = np.mean(scores_inner)
        if score > best_score:
            best_score = score
            best_c = c
    """
    train2=time.time()

    test=time.time()       
    #clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=best_c)
    clf = MultinomialNB()
    #clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    #clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    clf.fit(data_train, labels_train)
    #test_tfidf = vec.transform(data_test)#
    predicted = clf.predict(data_test)
    accuracy = metrics.accuracy_score(labels_test, predicted)
    precision = metrics.precision_score(labels_test, predicted, average='macro')
    recall = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kappa = metrics.cohen_kappa_score(labels_test, predicted)
    roc = metrics.roc_auc_score(labels_test.astype(int), predicted.astype(int))
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1_macro)
    kappas.append(kappa)
    scores_roc.append(roc)
    i += 1

stop = time.time()
print('Trainin time = '+str((train2 - train)/60))
print('Testing time = '+str((stop - test)/60))
    
#skf = StratifiedKFold(n_splits=10, random_state=3)
#scores_txt = cross_val_score(clf, corpus_tfidf, labels, scoring='f1_macro', cv=skf)
print('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies), np.std(accuracies) * 2))
print('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions), np.std(precisions) * 2))
print('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls), np.std(recalls) * 2))
print('F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print('Kappa: %0.2f (+/- %0.2f)' % (np.mean(kappas), np.std(kappas) * 2))
print('AUC: %0.2f (+/- %0.2f)' % (np.mean(scores_roc), np.std(scores_roc) * 2))
