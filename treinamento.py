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


def gridsearch(feature, sintoma, feature_name, grids, test_division):
    print(f"Test size: {test_division}")
    print(f"Feature set: {feature_name}")
    x_train = feature
    y_train = sintoma
    x_test = feature
    y_test = sintoma
    if test_division > 0.1:
        x_train, x_test, y_train, y_test = train_test_split(feature, sintoma, test_size=test_division, random_state=SEED, stratify=sintoma)
    
    for i, gs in enumerate(grids):
        print(f"\nEstimator: {grid_dict[i]}")
        gs.fit(x_train, y_train)
        print(f"Best params: {gs.best_params_}")
        print(f"Best training accuracy: {gs.best_score_}")
        results = gs.cv_results_
        cv_results = pd.DataFrame.from_dict(gs.cv_results_)
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.set_option("expand_frame_repr", False)
        result = cv_results[["params", "mean_test_Precision", "rank_test_Precision", "mean_test_F1", "rank_test_F1", "mean_test_Recal", "rank_test_Recal", "mean_test_Accuracy"]]
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        pd.set_option("expand_frame_repr", False)
        print(result)
        filename= f"{feature_name}_{sintoma.name}_{grid_dict[i]}_{test_division}.txt"
        with open(filename.replace("/","_"), 'w', encoding='utf-8') as f:
            f.write(f"Best params: {gs.best_params_}\n")
            f.write(f"Best training accuracy: {gs.best_score_}\n")
            f.write("\n\n")
            f.write(result.to_string())
        print("\n-----------------------------------------")
    print("\n\n----------------------------------------\n")

SEED = 42
CV = StratifiedKFold(10) #cross-validation
SCORE = 'accuracy'
SCORING = {'F1': make_scorer(f1_score), "Accuracy": make_scorer(accuracy_score), "Precision": make_scorer(precision_score), "Recal": make_scorer(recall_score)}

#Parameters for each method

PARAMS_RFC = [{'n_estimators':[100,120], 'criterion':['entropy', 'gini'], 'max_depth':[4, 6], 'min_samples_split':[2, 5, 10], 'min_samples_leaf':[1,2,4]}]

PARAMS_KNC = [{'n_neighbors':[1,10], 'leaf_size':[20,40,1], 'p':[1,2], 'weights':['uniform', 'distance'], 'metric':['minkowski', 'chebyshev']}]

PARAMS_SVC = [{'C':[0.1, 1, 10], 'gamma':[1, 0.1,], 'kernel':['sigmoid', 'rbf']}]

PARAMS_LR = [{'penalty':['l1', 'l2'], 'C': np.logspace(-4, 4, 50), 'tol': [0.0001]}]

PARAMS_DT = [{'criterion':['gini', 'entropy'], 'max_depth':[5,10,150]}]

PARAMS_NB = [{'var_smoothing': np.logspace(0, -9, num=100)}]

# Grid search for each method

GRID_RFC = GridSearchCV(
    estimator=RandomForestClassifier(random_state=SEED), 
    param_grid=PARAMS_RFC, 
    scoring=SCORING, 
    refit="F1", 
    cv=CV,
    return_train_score=True)

GRID_KNC = GridSearchCV(
    estimator=KNeighborsClassifier(), 
    param_grid=PARAMS_KNC, 
    scoring=SCORING,
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_SVC = GridSearchCV(
    estimator=SVC(random_state=SEED, probability=True), 
    param_grid=PARAMS_SVC, 
    scoring=SCORING, 
    refit="F1",
    cv=CV,
    return_train_score=True)

GRID_LR = GridSearchCV(
    estimator=LogisticRegression(random_state=SEED), 
    param_grid=PARAMS_LR, 
    scoring=SCORING,
    refit="F1", 
    cv=CV,
    return_train_score=True)

GRID_DT = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=SEED), 
    param_grid=PARAMS_DT, 
    scoring=SCORING, 
    refit="F1", 
    cv=CV, 
    return_train_score=True)

GRID_NB = GridSearchCV(
    estimator=GaussianNB(), 
    param_grid=PARAMS_NB, 
    scoring=SCORING, 
    refit="F1", 
    cv=CV,
    return_train_score=True)

# Grid list for easy iteration
grids = [GRID_SVC, GRID_DT, GRID_LR, GRID_NB]
grid_dict = { 0: "SVM", 1: "Decision Tree",  2: "Logistic Regression", 3: "Gaussian NB"}



facebook = pd.read_csv("teste.csv")

anew = pd.read_csv("anew.csv")
combined = pd.read_csv("combined.csv")
goemotions = pd.read_csv("goemotions.csv")
misc_features_facebook = pd.read_csv("misc_features_facebook.csv")
pos_facebook_sentences = pd.read_csv("pos_facebook_sentences.csv")
pos_facebook_sentences_spacy = pd.read_csv("pos_facebook_sentences_spacy.csv")
phq9 = pd.read_csv("facebook_phq9.csv")
liwc = pd.read_csv("facebook_liwc.csv")
tfidf = pd.read_csv("tfidf.csv")
tenses_facebook = pd.read_csv("tenses_facebook.csv")

features = {"Anew": anew, "GoEmotions": goemotions, "Misc_features_facebook": misc_features_facebook, "PoS_facebook_setences": pos_facebook_sentences, "PoS_facebook_sentences_spacy": pos_facebook_sentences_spacy, "PHQ9": phq9, "LIWC": liwc, "TFIDF": tfidf, "Tenses_facebook": tenses_facebook}
sintomas = ["Agitação/inquietação","Alteração de peso/apetite","Alteração de sono","Alteração na eficiência/funcionalidade","Cansaço/Desânimo/Desencorajamento/Fadiga/Perda de energia / Lentificação","Desamparo/Prejuízo social/Solidão","Desesperança","Desvalia / Baixa autoestima","Dificuldade para decidir","Déficit de atenção/Memória","Fator de risco","Fator protetivo, cuidado em saúde e bem-estar","Irritação / agressividade","Morte / Suicído de outro","Perda/Diminuição do prazer/ Perda/Diminuição da libido","Preocupação/Medo /Ansiedade","Sentimento de culpa","Sentimento de vazio","Tristeza/Humor depressivo","Suicído/Auto-extermínio","Sintoma físico"]

ppd = "Postagem com possível perfil depressivo"

for name, feature in features.items():
    for sintoma in sintomas:
        gridsearch(feature, facebook[sintoma], f"{name}", grids, 0.0)
        gridsearch(feature, facebook[sintoma], f"{name}", grids, 0.33)
        gridsearch(feature, facebook[sintoma], f"{name}", grids, 0.5)
        gridsearch(feature, facebook[sintoma], f"{name}", grids, 0.66)

#gridsearch(anew, sintoma, "ANEW", grids, 0.0)
#gridsearch(goemotions, sintoma, "GoEmotions", grids, 0.0)
#gridsearch(misc_features_facebook, sintoma, "misc_features_facebook", grids, 0.0)
#gridsearch(pos_facebook_sentences, sintoma, "pos_facebook_sentences", grids, 0.0)
#gridsearch(phq9, sintoma, "PHQ9", grids, 0.0)
#gridsearch(liwc, sintoma, "LIWC", grids, 0.0)
#gridsearch(tfidf, sintoma, "tfidf", grids, 0.0)
#gridsearch(pos_facebook_sentences_spacy, sintoma, "pos_facebook_sentences_spacy", grids, 0.0)
#gridsearch(tenses_facebook, sintoma, "tenses_facebook", grids, 0.0)

#gridsearch(anew, sintoma, "ANEW", grids, 0.33)
#gridsearch(goemotions, sintoma, "GoEmotions", grids, 0.33)
#gridsearch(misc_features_facebook, sintoma, "misc_features_facebook", grids, 0.33)
#gridsearch(pos_facebook_sentences, sintoma, "pos_facebook_sentences", grids, 0.33)
#gridsearch(phq9, sintoma, "PHQ9", grids, 0.33)
#gridsearch(liwc, sintoma, "LIWC", grids, 0.33)
#gridsearch(tfidf, sintoma, "tfidf", grids, 0.33)
#gridsearch(pos_facebook_sentences_spacy, sintoma, "pos_facebook_sentences_spacy", grids, 0.33)
#gridsearch(tenses_facebook, sintoma, "tenses_facebook", grids, 0.33)

#gridsearch(anew, sintoma, "ANEW", grids, 0.5)
#gridsearch(goemotions, sintoma, "GoEmotions", grids, 0.5)
#gridsearch(misc_features_facebook, sintoma, "misc_features_facebook", grids, 0.5)
#gridsearch(pos_facebook_sentences, sintoma, "pos_facebook_sentences", grids, 0.5)
#gridsearch(phq9, sintoma, "PHQ9", grids, 0.5)
#gridsearch(liwc, sintoma, "LIWC", grids, 0.5)
#gridsearch(tfidf, sintoma, "tfidf", grids, 0.5)
#gridsearch(pos_facebook_sentences_spacy, sintoma, "pos_facebook_sentences_spacy", grids, 0.5)
#gridsearch(tenses_facebook, sintoma, "tenses_facebook", grids, 0.5)

#gridsearch(anew, sintoma, "ANEW", grids, 0.66)
#gridsearch(goemotions, sintoma, "GoEmotions", grids, 0.66)
#gridsearch(misc_features_facebook, sintoma, "misc_features_facebook", grids, 0.66)
#gridsearch(pos_facebook_sentences, sintoma, "pos_facebook_sentences", grids, 0.66)
#gridsearch(phq9, sintoma, "PHQ9", grids, 0.66)
#gridsearch(liwc, sintoma, "LIWC", grids, 0.66)
#gridsearch(tfidf, sintoma, "tfidf", grids, 0.66)
#gridsearch(pos_facebook_sentences_spacy, sintoma, "pos_facebook_sentences_spacy", grids, 0.66)
#gridsearch(tenses_facebook, sintoma, "tenses_facebook", grids, 0.66)

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
for sintoma in sintomas:
    gridsearch(combined, facebook[sintoma], "Combined", grid_pipe, 0.33)
    gridsearch(combined, facebook[sintoma], "Combined", grid_pipe, 0.5)
    gridsearch(combined, facebook[sintoma], "Combined", grid_pipe, 0.66)
