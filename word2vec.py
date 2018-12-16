import glob

from janome.tokenizer import Tokenizer
from gensim.models import word2vec

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

# 品詞を取り出し「名詞、動詞、形容詞、形容動詞」のリスト作成
def tokenize(text):
    t = Tokenizer()
    tokens = t.tokenize(",".join(text))
    word = []
    for token in tokens:
        part_of_speech = token.part_of_speech.split(",")[0]
 
        if part_of_speech == "名詞":
            word.append(token.surface)        
        if part_of_speech == "動詞":        
            word.append(token.surface)
        if part_of_speech == "形容詞":
            word.append(token.surface)        
        if part_of_speech == "形容動詞":        
            word.append(token.surface)            
    return word

# ラベルと文章に分類
docs, labels = load_livedoor_news_corpus()
sentences = tokenize(docs[0:100])  # データ量が多いため制限している


# 以下に回答を作成してください
#word2vec.Word2Vecの引数に関して、リストにはsentencesを指定し、size=200, min_count=20, window=15としてください
model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
similar = model.most_similar(positive=["男"])
print(similar)
