import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

np.set_printoptions(precision=2)
docs = np.array([
    "リンゴ リンゴ", "リンゴ ゴリラ", "ゴリラ ラッパ"
])

# ベクトル表現に変換してください。
vectorizer = TfidfVectorizer(use_idf=True, token_pattern="(?u)\\b\\w+\\b")
vecs = vectorizer.fit_transform(docs)

print(vecs.toarray())
