from janome.tokenizer import Tokenizer
# 分かち書きをしてください
t = Tokenizer()
tokens = t.tokenize('すもももももももものうち', wakati=True)
print(tokens)
