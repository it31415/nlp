from janome.tokenizer import Tokenizer
# 形態素解析をしてください
tokenizer = Tokenizer()
tokens = tokenizer.tokenize("明日は晴れるだろうか。")
for token in tokens:
    print(token)
    print()
