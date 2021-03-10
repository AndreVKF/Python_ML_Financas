import re
import spacy
import string

def preprocessamento(texto, pln):
    texto = texto.lower()
    texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", " ", texto)
    texto = re.sub(r"https?://[A-Za-z0-9./]+", " ", texto)
    texto = re.sub(r" +", " ", texto)

    documento = pln(texto)
    lista = []
    for token in documento:
        lista.append(token.lemma_)

    lista = [palavra for palavra in lista if palavra not in pln.Defaults.stop_words and palavra not in string.punctuation] 
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

    return lista
