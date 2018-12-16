from gensim import corpora
from janome.tokenizer import Tokenizer


text1 = "すもももももももものうち"
text2 = "料理も景色もすばらしい"
text3 = "私の趣味は写真撮影です"

t = Tokenizer()
tokens1 = t.tokenize(text1, wakati=True)
tokens2 = t.tokenize(text2, wakati=True)
tokens3 = t.tokenize(text3, wakati=True)

documents = [tokens1, tokens2, tokens3]
print(documents)
# corporaを使い単語辞書を作成してください。
dictionary = corpora.Dictionary(documents)
print(dictionary)
# 各単語のidを表示してください
print(dictionary.token2id)

# Bag of Wordsの作成してください
bow_corpus = [dictionary.doc2bow(d) for d in documents]

# (id, 出現回数)のリストが出力されます。
print(bow_corpus)

print()
# bow_corpusの内容をわかりやすく出力する
texts = [text1, text2, text3]
for i in range(len(bow_corpus)):
    print(texts[i])
    for j in range(len(bow_corpus[i])):
        index = bow_corpus[i][j][0]
        num = bow_corpus[i][j][1]
        print("\"", dictionary[index], "\" が " ,num, "回", end=", ")
    print()
