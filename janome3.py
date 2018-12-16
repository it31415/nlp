from janome.tokenizer import Tokenizer
t = Tokenizer()
tokens = t.tokenize("豚の肉を食べた")
# 以下のリストに答えを代入してください
word = []
# 以下に回答を作成してください
for token in tokens:
   part_of_speech = token.part_of_speech.split(',')[0] 
    
   if part_of_speech == '名詞' or part_of_speech == '動詞':
       word.append(token.surface)
    
print(word)
