try:
    import stanza
except:
    print("stanza is not installed. If you want to use the stanza tokenizer, please install the package.")
    pass

STOPWORDS = ["SF", "SE", "SS", "SP", "SO", "SW", "SH", "SL", "SN", "NF", "NV", "NA"]
FUNCTIONWORDS = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX", "EP", "EF", "EC", "ETN", "ETM",
                         "XPN", "XSN", "XSV", "XSA"]

def remove_pos(token_pos_tuple, pos_list):
    """
    Remove given POSs in the tokenized tuple
    :param token_pos_tuple: list of tuple consisting of token and POS [('열심히', 'Adverb'), ('코딩', 'Noun')...]
    :param pos_list: list of POS to be removed
    :return: cleaned tuple of ('token', 'Part-Of-Speech') as list
    """
    pos_tuple_cleaned = []

    for index, pair in enumerate(token_pos_tuple):
        if pair[1] not in pos_list:
            pos_tuple_cleaned.append(pair)

    return pos_tuple_cleaned


def tokenize(tokenizer, text):
    """
    tokenize sequences using konlpy tokenizer.
    :param tokenizer: str, possible options: (okt, komoran, mecab, kkma, hannanum, stanza)
    :param text: str, raw text
    :return: tuple (pos_tuple_all, pos_tuple_cleaned, tokens_cleaned)
            where pos_tuple_all consists of tuple ('token', 'Part-Of-Speech') of all raw tokens (including stopwords like punctuation, numbers, URL ...)
                  pos_tuple_cleaned consists of tuple ('token', 'Part-Of-Speech') of contents words (+ function words if the param include_function_words=True)
                  tokens_cleaned is a list of stopword removed tokens (if the param include_function_words=True, function words are also included)
    """

    if tokenizer == 'stanza-custom':
        tagger = stanza.Pipeline(lang='ko', package='gsd',
                  pos_model_path='./custom-model/pos/ko_gsd_tagger.pt',
                  lemma_model_path = './custom-model/lemma/ko_gsd_lemmatizer.pt',
                  depparse_model_path ='./custom-model/depparse/ko_gsd_parser.pt')
        stopwords = STOPWORDS
    else:
        raise ValueError("tokenizer must be stanza-custom")

    # tokenize
    if tokenizer == 'stanza-custom':
        doc = tagger(text)
        pos_tuple_all = [
            (word.lemma.split('+'), word.xpos.split('+'))
            for sent in doc.sentences
            for word in sent.words
        ]
    else:
        raise ValueError("tokenizer must be 'stanza-custom'")

    # remove stopwords
    pos_tuple_cleaned = remove_pos(pos_tuple_all, pos_list=stopwords)

    # separate lists for tokens
    tokens_cleaned = [item[0] for item in pos_tuple_cleaned]

    return pos_tuple_all, pos_tuple_cleaned, tokens_cleaned


def remove_function_words(pos_tuple, tokenizer):
    """
    Remove function words from tokenized pos tuple
    include this if main argument functionwords=False
    :param pos_tuple: ('token', 'Part-Of-Speech')
    :param tokenizer: str, available options: okt, komoran, mecab, kkma, hannanum
    :return:
    """

    functionwords = []

    if tokenizer == 'stanza-custom':
        functionwords = FUNCTIONWORDS

    pos_tuple_cleaned = remove_pos(pos_tuple, pos_list=functionwords)

    # separate lists for tokens
    tokens_cleaned = [item[0] for item in pos_tuple_cleaned]

    return pos_tuple_cleaned, tokens_cleaned
