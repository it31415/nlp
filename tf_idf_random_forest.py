import glob
import random
from sklearn.model_selection import train_test_split
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


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
                url = lines[0]  
                datetime = lines[1]  
                subject = lines[2]
                body = "".join(lines[3:])
                text = subject + body

            docs.append(text)
            labels.append(c_id)

    return docs, labels

docs, labels = load_livedoor_news_corpus()

# データをトレイニングデータとテストデータに分割する（機械学習概論 「ホールドアウト法の理論と実践」)
train_data, test_data, train_labels, test_labels = train_test_split(docs, labels, test_size=0.2, random_state=0)

# tf-idfでトレイニングデータとテストデータをベクトル化する。(「fit関数」)
# 以下に回答を作成してください
#-------------------------------------------------------

vectorizer = TfidfVectorizer()
train_matrix = vectorizer.fit_transform(train_data) # train_dataをベクトル化
test_matrix = vectorizer.transform(test_data)# test_dataをベクトル化

#-------------------------------------------------------


# ランダムフォレストで学習（教師あり分類　「ランダムフォレスト」)
clf = RandomForestClassifier(n_estimators=2)
clf.fit(train_matrix, train_labels)



# 精度の出力
print(clf.score(train_matrix, train_labels))
print(clf.score(test_matrix, test_labels))
