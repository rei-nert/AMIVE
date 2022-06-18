#! /usr/bin/env python3

features = ["Anew", "GoEmotions", "Misc_features_facebook", "PoS_facebook_setences", "PoS_facebook_sentences_spacy", "PHQ9", "LIWC", "TFIDF", "Tenses_facebook", "Combined"]
sintomas = ["Agitação/inquietação","Alteração de peso/apetite","Alteração de sono","Alteração na eficiência/funcionalidade","Cansaço/Desânimo/Desencorajamento/Fadiga/Perda de energia / Lentificação","Desamparo/Prejuízo social/Solidão","Desesperança","Desvalia / Baixa autoestima","Dificuldade para decidir","Déficit de atenção/Memória","Fator de risco","Fator protetivo, cuidado em saúde e bem-estar","Irritação / agressividade","Morte / Suicído de outro","Perda/Diminuição do prazer/ Perda/Diminuição da libido","Preocupação/Medo /Ansiedade","Sentimento de culpa","Sentimento de vazio","Tristeza/Humor depressivo","Suicído/Auto-extermínio","Sintoma físico"]
sizes = ["0.0", "0.33", "0.5", "0.66"]
grids = ["SVM", "Decision Tree", "Logistic Regression", "Gaussian NB"]


results = {}

for sintoma in sintomas:
    for feature in features:
        for size in sizes:
            params = ""
            accuracy = 0.0
            file = ""
            if feature == "Combined" and size == "0.0":
                continue
            for grid in grids:
                filename = f"{feature}_{sintoma}_{grid}_{size}.txt"
                filename = filename.replace("/", "_")
                with open(f"results_binary/{filename}", 'r', encoding='utf-8') as f:
                    firstline = f.readline().strip().split(": ", 1)
                    secondline = f.readline().strip().split(": ", 1)

                read_param = firstline[1]
                read_accuracy = round(float(secondline[1]), 5)
                if read_accuracy > accuracy:
                    accuracy = read_accuracy
                    params = read_param
                    file = filename
            temp_tupple = (feature, sintoma, size)
            results[temp_tupple] = [accuracy, params, file]

for feature in features:
    print(feature, ": ")
    for sintoma in sintomas:
        print("\t",sintoma, ": ")
        for size in sizes:
            if feature == "Combined" and size == "0.0":
                continue
            print("\t\t", size, ": ")
            temp_tupple = (feature, sintoma, size)
            print("\t\t\t", results[temp_tupple])

