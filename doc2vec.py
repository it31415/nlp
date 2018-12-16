import glob
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer


# livedoor newsの読み込みと分類
def load_livedoor_news_corpus():
    category = {
        "dokujo-tsushin": 1,
        "it-life-hack":2,
        "kaden-channel": 3,
        "livedoor-homme": 4,
        "movie-enter": 5,
        "peachy": 6,
        "smax": 7,
        "sports-watch": 8,
        "topic-news":9
    }
    docs  = []
    labels = []

    for c_name, c_id in category.items():
        files = glob.glob("./text/{c_name}/{c_name}*.txt".format(c_name=c_name))

        text = ""
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines() 

                #1,2行目に書いたあるURLと時間は関係ないので取り除きます。
                url = lines[0]  
                datetime = lines[1]  
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels
docs, labels = load_livedoor_news_corpus()

# Doc2Vecの処理
token = [] # 各docsの分かち書きした結果を格納するリストです
training_docs = [] # TaggedDocumentを格納するリストです
for i in range(4):
    
    # docs[i] を分かち書きして、tokenに格納します
    t = Tokenizer() 
    token.append(t.tokenize(docs[i], wakati=True))
    
    # TaggedDocument クラスのインスタンスを作成して、結果をtraining_docsに格納します
    # タグは "d番号"とします
    training_docs.append(TaggedDocument(words=token[i], tags=["d" + str(i)]))

print(len(token))
print(len(training_docs))
# 以下に回答を作成してください
#-------------------------------------------------------
model = Doc2Vec(documents=training_docs, min_count=1)

#-------------------------------------------------------

for i in range(4):
    print(model.docvecs.most_similar("d"+str(i)))
