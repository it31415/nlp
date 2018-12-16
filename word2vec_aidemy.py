from gensim.models import word2vec
from janome.tokenizer import Tokenizer
from aidemy_document import *

t = Tokenizer()
tokens = t.tokenize(text)
word=[]

for token in tokens:
	part_of_speech = token.part_of_speech.split(",")[0]

	if part_of_speech == "名詞":
	    word.append(token.surface)        
	#if part_of_speech == "動詞":        
	#    word.append(token.surface)
	#if part_of_speech == "形容詞":
	#    word.append(token.surface)        
	#if part_of_speech == "形容動詞":        
	#    word.append(token.surface)            

documents = [word]
model = word2vec.Word2Vec(documents, size=200, min_count=3, window=15)
similar = model.most_similar(positive=["機械", "学習"], topn = 20)
print(similar)
