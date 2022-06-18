#!/usr/bin/env python3

# imports of all needed tools
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, log_loss, make_scorer, classification_report, recall_score, f1_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain

def multilabel_search(feature, sintoma, grids, test_division):
    print(f"Test size: {test_division}")
    x_train, x_test, y_train, y_test = train_test_split(feature, sintoma, test_size=test_division, random_state=SEED)
    gs = GridSearchCV(BinaryRelevance(), grids, scoring='accuracy')
    gs.fit(x_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best training accuracy: {gs.best_score_}")
#    result = cv_results[["params", "mean_test_Precision", "rank_test_Precision", "mean_test_F1", "rank_test_F1", "mean_test_Recal", "rank_test_Recal", "mean_test_Accuracy"]]
#    pd.set_option("display.max_rows", None, "display.max_columns", None)
#    pd.set_option("expand_frame_repr", False)
#    print(result)
    
#def gridsearch(feature, sintoma, feature_name, grids, test_division):
#    print(f"Test size: {test_division}")
#    print(f"Feature set: {feature_name}")
#    x_train, x_test, y_train, y_test = train_test_split(feature, sintoma, test_size=test_division, random_state=SEED)
#    
#    for i, gs in enumerate(grids):
#        print(f"\nEstimator: {grid_dict[i]}")
#        grid = grid_dict[i]
#        fitted= gs.fit(x_train, y_train)
#        predicted = fitted.predict(x_test)
#        print(f"Best params: {gs.best_params_}")
#        print(f"Best training accuracy: {gs.best_score_}")
#        results = gs.cv_results_
#        cv_results = pd.DataFrame.from_dict(gs.cv_results_)
#        pd.set_option("display.max_rows", None, "display.max_columns", None)
#        pd.set_option("expand_frame_repr", False)
#        result = cv_results[["params", "mean_test_Precision", "rank_test_Precision", "mean_test_F1", "rank_test_F1", "mean_test_Recal", "rank_test_Recal", "mean_test_Accuracy"]]
#        pd.set_option("display.max_rows", None, "display.max_columns", None)
#        pd.set_option("expand_frame_repr", False)
#        print(result)
#        print(gs.predict_prob(x_test))
#
##            filename= f"{feature_name}_{sintoma}_{grid_dict[i]}_{test_division}.txt"
# #           with open(filename.replace("/","_"), 'w', encoding='utf-8') as f:
# #               f.write(f"Estimator: {grid}\n")
# #               f.write(f"Sintoma: {sintoma.name}\n")
# #               f.write(f"Best params: {gs.best_params_}\n")
# #               f.write(f"Best training accuracy: {gs.best_score_}\n")
# #               f.write("\n\n")
# #               f.write(result.to_string())
#        print("\n-----------------------------------------")
#    print("\n\n----------------------------------------\n")

SEED = 42
CV = StratifiedKFold(10) #cross-validation
#CV = None
SCORING = {'F1': make_scorer(f1_score), "Accuracy": make_scorer(accuracy_score), "Precision": make_scorer(precision_score), "Recal": make_scorer(recall_score)}
COUNT_VEC = TfidfVectorizer()
OHE = OneHotEncoder(handle_unknown='ignore')

#Parameters for each method

PARAMS_RFC = {'classifier': [RandomForestClassifier()],'classifier__n_estimators':[100,120], 'classifier__criterion':['entropy', 'gini'], 'classifier__max_depth':[4, 6], 'classifier__min_samples_split':[2, 5, 10], 'classifier__min_samples_leaf':[1,2,4]}

PARAMS_KNC = {'classifier': [KNeighborsClassifier()], 'classifier__n_neighbors':[1,10], 'classifier__leaf_size':[20,40,1], 'classifier__p':[1,2], 'classifier__weights':['uniform', 'distance'], 'classifier__metric':['minkowski', 'chebyshev']}

PARAMS_SVC = {'classifier': [SVC()], 'classifier__C':[0.1, 1, 10], 'classifier__gamma':[1, 0.1,], 'classifier__kernel':['sigmoid', 'rbf']}

PARAMS_LR = {'classifier':[LogisticRegression()], 'classifier__penalty':['l1', 'l2', 'none'], 'classifier__C': np.logspace(-4, 4, 50), 'classifier__tol': [0.0001], 'classifier__solver': ['liblinear', 'lbfgs']}

PARAMS_DT = {'classifier': [DecisionTreeClassifier()],'classifier__criterion':['gini', 'entropy'], 'classifier__max_depth':[5,10,150]}

PARAMS_NB = {'classifier':[GaussianNB()], 'classifier__var_smoothing': np.logspace(0, -9, num=100)}

grids = [PARAMS_RFC, PARAMS_KNC, PARAMS_SVC, PARAMS_LR, PARAMS_DT, PARAMS_NB]

# Grid search for each method

#GRID_RFC = GridSearchCV(
#    estimator=RandomForestClassifier(random_state=SEED), 
#    param_grid=PARAMS_RFC, 
#    scoring=SCORING, 
#    refit="F1", 
#    cv=CV,
#    return_train_score=True)
#
#GRID_KNC = GridSearchCV(
#    estimator=KNeighborsClassifier(), 
#    param_grid=PARAMS_KNC, 
#    scoring=SCORING,
#    refit="F1",
#    cv=CV,
#    return_train_score=True)
#
#GRID_SVC = GridSearchCV(
#    estimator=SVC(random_state=SEED, probability=True), 
#    param_grid=PARAMS_SVC, 
#    scoring=SCORING, 
#    refit="F1",
#    cv=CV,
#    return_train_score=True)
#
#GRID_LR = GridSearchCV(
#    estimator=LogisticRegression(random_state=SEED), 
#    param_grid=PARAMS_LR, 
#    scoring=SCORING,
#    refit="F1", 
#    cv=CV,
#    return_train_score=True)
#
#GRID_DT = GridSearchCV(
#    estimator=DecisionTreeClassifier(random_state=SEED), 
#    param_grid=PARAMS_DT, 
#    scoring=SCORING, 
#    refit="F1", 
#    cv=CV, 
#    return_train_score=True)
#
#GRID_NB = GridSearchCV(
#    estimator=GaussianNB(), 
#    param_grid=PARAMS_NB, 
#    scoring=SCORING, 
#    refit="F1", 
#    cv=CV,
#    return_train_score=True)
#
## Grid list for easy iteration
#grids = [GRID_SVC, GRID_DT, GRID_LR, GRID_NB]
#grid_dict = { 0: "SVM", 1: "Decision Tree",  2: "Logistic Regression", 3: "Gaussian NB"}
#
facebook = pd.read_csv("teste.csv")
text = facebook["text"]
X_set = COUNT_VEC.fit_transform(text).toarray()


#anew = pd.read_csv("anew.csv")
#combined = pd.read_csv("combined.csv")
#goemotions = pd.read_csv("goemotions.csv")
#misc_features_facebook = pd.read_csv("misc_features_facebook.csv")
#pos_facebook_sentences = pd.read_csv("pos_facebook_sentences.csv")
#pos_facebook_sentences_spacy = pd.read_csv("pos_facebook_sentences_spacy.csv")
#phq9 = pd.read_csv("facebook_phq9.csv")
#liwc = pd.read_csv("facebook_liwc.csv")
#tfidf = pd.read_csv("tfidf.csv")
#tenses_facebook = pd.read_csv("tenses_facebook.csv")
#
#features = {"Anew": anew, "GoEmotions": goemotions, "Misc_features_facebook": misc_features_facebook, "PoS_facebook_setences": pos_facebook_sentences, "PoS_facebook_sentences_spacy": pos_facebook_sentences_spacy, "PHQ9": phq9, "LIWC": liwc, "TFIDF": tfidf, "Tenses_facebook": tenses_facebook}
sintomas = ["Agitação/inquietação","Alteração de peso/apetite","Alteração de sono","Alteração na eficiência/funcionalidade","Cansaço/Desânimo/Desencorajamento/Fadiga/Perda de energia / Lentificação","Desamparo/Prejuízo social/Solidão","Desesperança","Desvalia / Baixa autoestima","Dificuldade para decidir","Déficit de atenção/Memória","Fator de risco","Fator protetivo, cuidado em saúde e bem-estar","Irritação / agressividade","Morte / Suicído de outro","Perda/Diminuição do prazer/ Perda/Diminuição da libido","Preocupação/Medo /Ansiedade","Sentimento de culpa","Sentimento de vazio","Tristeza/Humor depressivo","Suicído/Auto-extermínio","Sintoma físico"]

ppd = "Postagem com possível perfil depressivo"

name = "Facebook"
#new_tags = dict()
#for sintoma in sintomas:
#    for index, row in facebook.iterrows():
#        if row[sintoma]:
#            list_sintomas = new_tags.get(index, [])
#            list_sintomas.append(sintoma)
#            new_tags[index] = list_sintomas
#        else:
#            list_sintomas = new_tags.get(index, [])
#            new_tags[index] = list_sintomas
#
#new_y = [("",)] * len(new_tags)
#
#for key, item in new_tags.items():
#    new_y[key] =  tuple(item)
#reshape_new_y = np.array(new_y).reshape(-1,1)
#new_y_ohe = OHE.fit_transform(reshape_new_y).toarray()

multilabel_search(X_set, facebook[sintomas], grids, 0.33)
multilabel_search(X_set, facebook[sintomas], grids, 0.5)
multilabel_search(X_set, facebook[sintomas], grids, 0.66)
#gridsearch(X_set, facebook[sintomas], grids, 0.33)
#gridsearch(X_set, facebook[sintomas], f"{name}", grids, 0.5)
#gridsearch(X_set, facebook[sintomas], f"{name}", grids, 0.66)


#-------------------PIPELINES ------------------

PIPE_GRID_PARAMS_RFC = [{
'clf__n_estimators': [100,120],
'clf__criterion': ['entropy','gini'],
'clf__max_depth': [4, 6],
'clf__min_samples_split': [2,5,10],
'clf__min_samples_leaf': [1, 2, 4]}]

PIPE_GRID_PARAMS_KNC = [{
'clf__n_neighbors': [1,10],
'clf__leaf_size': [20,40,1],
'clf__p': [1,2],
'clf__weights': ['uniform', 'distance'],
'clf__metric': ['minkowski', 'chebyshev']}]

PIPE_GRID_PARAMS_SVC = [{
'clf__C': [0.1, 1, 10],
'clf__gamma': [1, 0.1,],
'clf__kernel': ['sigmoid', 'rbf']}]

PIPE_GRID_PARAMS_LR = [{
'clf__penalty' : ['l1', 'l2'],
'clf__C': np.logspace(-4, 4, 50),
'clf__tol' : [0.0001]}]

PIPE_GRID_PARAMS_DT = [{
'clf__criterion':['gini','entropy'],
'clf__max_depth':[5,10,150]}]

PIPE_GRID_PARAMS_NB = [{
'clf__var_smoothing': np.logspace(0,-9, num=100)}]

PIPE_GRID_RFC = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', RandomForestClassifier(random_state=SEED))])

PIPE_GRID_KNC = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', KNeighborsClassifier())])

PIPE_GRID_SVC = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', SVC(random_state=SEED, probability=True))])

PIPE_GRID_LR = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', LogisticRegression(random_state=SEED))])

PIPE_GRID_DT = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', DecisionTreeClassifier(random_state=SEED))])

PIPE_GRID_NB = Pipeline([('feature_select', SelectPercentile(chi2)), ('clf', GaussianNB())])

GRID_PIPE_RFC = GridSearchCV(
    estimator=PIPE_GRID_RFC,
    param_grid=PIPE_GRID_PARAMS_RFC,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_PIPE_KNC = GridSearchCV(
    estimator=PIPE_GRID_KNC,
    param_grid=PIPE_GRID_PARAMS_KNC,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)


GRID_PIPE_SVC = GridSearchCV(
    estimator=PIPE_GRID_SVC,
    param_grid=PIPE_GRID_PARAMS_SVC,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_PIPE_LR = GridSearchCV(
    estimator=PIPE_GRID_LR,
    param_grid=PIPE_GRID_PARAMS_LR,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_PIPE_DT = GridSearchCV(
    estimator=PIPE_GRID_DT,
    param_grid=PIPE_GRID_PARAMS_DT,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_PIPE_NB = GridSearchCV(
    estimator=PIPE_GRID_NB,
    param_grid=PIPE_GRID_PARAMS_NB,
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

grid_pipe = [GRID_PIPE_SVC, GRID_PIPE_DT, GRID_PIPE_LR, GRID_PIPE_NB]
#gridsearch(X_set, facebook[sintoma], "Facebook pipeline", grid_pipe, 0.33)
#gridsearch(X_set, facebook[sintoma], "Facebook pipeline", grid_pipe, 0.5)
#gridsearch(X_set, facebook[sintoma], "Facebook pipeline", grid_pipe, 0.66)
