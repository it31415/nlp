from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

docs = ["これはペンです", "私はあほです", "俺は男です", "あなたはサルです"]

token = [] # 各docsの分かち書きした結果を格納するリストです
training_docs = [] # TaggedDocumentを格納するリストです

for i in range(4):
    
    # docs[i] を分かち書きして、tokenに格納します
    t = Tokenizer() 
    token.append(t.tokenize(docs[i], wakati=True))
    
    # TaggedDocument クラスのインスタンスを作成して、結果をtraining_docsに格納します
    # タグは "d番号"とします
    training_docs.append(TaggedDocument(words=token[i], tags=["d" + str(i)]))

# 以下に回答を作成してください
#-------------------------------------------------------
model = Doc2Vec(documents=training_docs, min_count=1)

#-------------------------------------------------------

for i in range(4):
    print(model.docvecs.most_similar("d"+str(i)))
