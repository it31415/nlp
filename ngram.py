from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("太郎はこの本を二郎を見た女性に渡した。", wakati=True)

def gen_Ngram(words,N):
    # Ngramを生成してください
    ngram = []
    for i in range(len(words)-N+1):
        cw = ''.join(words[i:i+N])
        ngram.append(cw)

    return ngram

print(gen_Ngram(tokens, 2))
print(gen_Ngram(tokens, 3))
