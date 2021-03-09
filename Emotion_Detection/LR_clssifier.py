import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


def getTfidfVectors(dataset_path):
    df = pd.read_csv(dataset_path)
    text = df["text"]
    label = df["labels"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    pickle.dump(vectorizer, open("tfidf_vectorizer.pkl","wb"))
    X = vectorizer.transform(text)
    return X,label

def predict(text, name = "SVM"):
    vectorier = pickle.load(open("tfidf_vectorizer.pkl","rb"))
    vectors = vectorier.transform([text]).todense()
    model = pickle.load(open(name+".pkl", "rb"))
    prediction = model.predict(vectors)[0]
    return prediction



def printTfidfResults(classifier,classifier_name,dataset_path):
    vectors, labels = getTfidfVectors(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.3, random_state=42)
    
    print(" TFIDF With "+classifier_name)
    classifier = classifier.fit(X_train,y_train)

    pickle.dump(classifier, open(classifier_name+".pkl", 'wb'))

    probs = classifier.predict_proba(X_test)
    y_pred = classifier.predict(X_test)
    print(classifier_name + " TFIDF " + 'accuracy %s' % accuracy_score(y_pred, y_test))
    print(classifier_name+" TFIDF "+' precision weighted %s' % precision_score(y_pred, y_test, average="weighted"))
    print(classifier_name+" TFIDF "+' recall weighted%s' % recall_score(y_pred, y_test, average="weighted"))
    print(classifier_name+" TFIDF "+' f1-score weighted%s' % f1_score(y_pred, y_test, average="weighted"))

    # with open('keras_encoding.csv', 'a', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow([classifier_name+" Keras Encoding "+', %s' % precision_score(y_pred, y_test, average="macro")+
    #       ' , %s' % precision_score(y_pred, y_test, average="micro")+
    #     ', %s' % precision_score(y_pred, y_test, average="weighted")+', %s' % recall_score(y_pred, y_test, average="macro")
    #     +', %s' % recall_score(y_pred, y_test, average="micro")
    #     +', %s' % recall_score(y_pred, y_test, average="weighted")
    #     +', %s' % f1_score(y_pred, y_test, average="macro")
    #     +', %s' % f1_score(y_pred, y_test, average="micro")
    #     +', %s' % f1_score(y_pred, y_test, average="weighted")])

    return probs
    
dataset_path="C:/Users/Pak/OneDrive/Desktop/Emotion_Detection/dataset.csv"
# lgr = LogisticRegression()
# lg_probs=printTfidfResults(lgr,"GaussianProcessClassifier",dataset_path)

# Svm = svm.SVC(probability=True)
# svm_probs=printTfidfResults(Svm,"SVM",dataset_path)



