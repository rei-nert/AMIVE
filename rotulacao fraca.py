import json,spacy,pandas,sys

#folder = sys.argv[1]
dest = sys.argv[1]
sp = spacy.load("pt_core_news_sm")

#no caso teria que adaptar para usar o csv que tá no drive ao invés de um arquivo json.
with open("lexico/signs.json",'rb') as f:
    r = f.read()
termos = json.loads(r)
print (termos)

max = 0
sintomas = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
for sintoma in sintomas:
    for termo in termos[sintoma]:
        aux = len(termo.split("_")) - 1
        if max < aux:
            max = aux
#número máximo de n_gramas é 6.
print(max)
#um dataframe por sintoma
dataframes = [pandas.DataFrame() for i in range(10)]


#PARA POSTAGENS DO CORPUS DO IVANDRÉ:
data = pandas.read_csv("normalised_facebook.csv")
posts = data['text'].to_numpy()

#provavelmente tem um jeito bem mais elegante de fazer a mesma coisa
#realmente usar expressão regular ou até mesmo um .find é bem melhor. Não sei onde eu tava com a cabeça

for index,post in data.iterrows():
    #esses provavelmente não são os nomes
    unigrams = []
    bigrams  = []
    trigrams = []
    quadgrams = []
    pentagrams = []
    hexagrams = []

    #usando o spacy para tokenizar
    tokens = sp(post['text'])
    for token in tokens:
        unigrams.append(token.text)

    for i, token in enumerate(tokens[:-1]):
        bigrams.append(token.text+"_"+tokens[i+1].text)

    for i, token in enumerate(tokens[:-2]):
        trigrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text)

    for i, token in enumerate(tokens[:-3]):
        quadgrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text+"_"+tokens[i+3].text)

    for i, token in enumerate(tokens[:-4]):
        pentagrams.append(token.text+"_"+tokens[i+1].text+"_"+tokens[i+2].text+"_"+tokens[i+3].text+tokens[i+4].text)

    for i, token in enumerate(tokens[:-5]):
        hexagrams.append(token.text+"_"+tokens[i+1].text +"_"+tokens[i+2].text+"_"+tokens[i+3].text+"_" + tokens[i+4].text + "_" + tokens[i+5].text)
    unigrams = set(unigrams)
    bigrams = set(bigrams)
    trigrams = set(trigrams)
    quadgrams = set(quadgrams)
    pentagrams = set(pentagrams)
    hexagrams = set(hexagrams)

    for i,sintoma in enumerate(sintomas):
        terms = set(termos[sintoma])
        #if not unigrams.isdisjoint(terms) or not bigrams.isdisjoint(terms) or not trigrams.isdisjoint(terms) or not quadgrams.isdisjoint(terms) or not pentagrams.isdisjoint(terms) or not hexagrams.isdisjoint(terms):
        post_terms  = unigrams.intersection(terms)  
        post_terms.update(bigrams.intersection(terms))
        post_terms.update(trigrams.intersection(terms))
        post_terms.update(quadgrams.intersection(terms))
        post_terms.update(pentagrams.intersection(terms))
        post_terms.update(hexagrams.intersection(terms))
        if post_terms:
            row = {"texto":post['text']}
            #.append está deprecado, e fica gerando warnings. Mudar para .concat 
            dataframes[i] = dataframes[i].append(row,ignore_index=True)

for i,dataframe in enumerate(dataframes):
    filename = f"{dest}/sintoma_" + str(i) +".csv"
    with open(filename,"wb") as f:
        pass
    dataframe.to_csv(filename)