TOKENS = {
    "UNK": 1,
    "EOS": 2,
    "START": 3,
    "END": 4,
    "CHILD": 5,
    "PARENT": 6,
    "PAD": 0,
}

NA = 0

EXOPHOR = {
    "著者": 1,
    "読者": 2,
    "不特定:人": 3,  # 不特定:人１, 不特定:人２
    # "不特定:物": 4,
    # "不特定:状況": 5,
}

CASES = {
    "ガ２": 0,
    "ガ": 1,
    "ヲ": 2,
    "ニ": 3,
    # "ノ": 4,
    "coref": 4,
}

DEP_TYPES = {
    "na": 0,
    "dep": 1,
    "intra": 2,
    "inter": 3,
    "exo": 4,
    "overt": 5,
}

BP_TYPES = {
    "PAD": 0,
    "EOS": 1,
    "NP": 2,
    "PP": 3,
}

MRPH_INFO = {
    "word_id": 0,
    "pos_id": 1,
    "spos_id": 2,
    "conj_id": 3,
}

FEATURES = [
    "PathEmbedding",
    "StringMatch",
    "SentenceDistance",
    "SelectionalPreference",
    "SynonymDictionary"
]
