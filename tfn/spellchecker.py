from spellchecker import SpellChecker
import pickle_file

spell = SpellChecker(distance=1)
def check_spelling(tokens, keep_wrong=False):
    if keep_wrong:
        length_original = len(tokens)
        tokens += [
            spell.correction(token) for token in tokens
            if not spell.correction(token) in [
                token for token in tokens
            ]
        ]
        return tokens, len(tokens) - length_original

    elif not keep_wrong:
        corrections = [
            (token, spell.correction(token)) for token in tokens
            if not token == spell.correction(token)
        ]
        for correction in corrections:
            tokens.remove(correction[0])
            tokens.append(correction[1])

        #return tokens, len(corrections)
        return len(corrections)     


def check_spell(data, data_type):
    output = []
    len_correction = []
    for text in data:
        if not text:
            continue
        output.append(check_spelling(text, keep_wrong=False))[0]
        len_correction(check_spelling(text, keep_wrong=False))[1]

    print('The length of {} is{}'.format(data_type, len_correction))

# Preprocessed data using NLTK stemming
data_stemmer = pickle_file.load("books_stemmer.pkl")
data_stemmer = data_stemmer['contents'].tolist()

# Preprocessed data using NLTK lemmatization
data_lemmatization = load("books_lemmatization.pkl")
data_lemmatization = data_lemmatization['contents'].tolist()

# Preprocessed data with the above two 
data = pd.read_pickle("books.pkl")
data = data['contents'].tolist()

check_spell(data_stemmer, 'Stemming')
check_spell(data_lemmatization, 'Lemmatization')
check_spell(data, 'None')
