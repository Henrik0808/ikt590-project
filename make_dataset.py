from sklearn.datasets import fetch_20newsgroups
import re

if __name__ == '__main__':

    categories = ('rec.sport.baseball', 'rec.sport.hockey', 'alt.atheism', 'soc.religion.christian', 'talk.religion.misc')
    category_id_to_name = lambda x: 'sports' if x < 2 else 'religion'

    Xs, Ys = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        return_X_y=True
    )

    PATTERN_SENTENCE_END = re.compile(r'[.!?]+')
    PATTERN_WHITESPACE = re.compile(r'\s+')
    PATTERN_NOISE = re.compile(r'[,"()[\]:^<>*~_|#{}+]+')
    ignore_sequences = ('---', '==', '\\\\', '//', '@')

    clean_10 = set()
    clean_11 = set()
    for X, Y in zip(Xs, Ys):
        sentences = PATTERN_SENTENCE_END.split(X)
        sentences = [PATTERN_NOISE.sub('', PATTERN_WHITESPACE.sub(' ', sentence)).strip() for sentence in sentences]
        sentences = [sentence.lower() for sentence in sentences if sentence != '']

        for sentence in sentences:
            if any(sequence in sentence for sequence in ignore_sequences):
                continue

            word_count = len(sentence.split())
            if word_count == 10:
                clean_10.add((sentence, Y))
            elif word_count == 11:
                clean_11.add((sentence, Y))

    sum1, sum2 = 0, 0
    for _, Y in clean_10:
        cat = category_id_to_name(Y)
        if cat == 'sports':
            sum1 += 1
        else:
            sum2 += 1
    print(f'words_10.csv, sports: {sum1}, religion: {sum2}')
    
    sum1, sum2 = 0, 0
    for _, Y in clean_11:
        cat = category_id_to_name(Y)
        if cat == 'sports':
            sum1 += 1
        else:
            sum2 += 1
    print(f'words_11.csv, sports: {sum1}, religion: {sum2}')

    with open('words_10.csv', mode='w', encoding='utf-8') as f:
        for X, Y in clean_10:
            f.write(category_id_to_name(Y) + ',' + X + '\n')
    with open('words_11.csv', mode='w', encoding='utf-8') as f:
        for X, Y in clean_11:
            f.write(category_id_to_name(Y) + ',' + X + '\n')
