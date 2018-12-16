import glob

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
    
    # 以下に説明文のコードの該当箇所を写してください
    #-----------------------------------
    for c_name, c_id in category.items():
       files = glob.glob("./text/{c_name}/{c_name}*.txt".format(c_name=c_name))
       print("category: ", c_name, ", ", len(files))
       
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
    #-----------------------------------

docs, labels = load_livedoor_news_corpus()

print("\nlabel: ", labels[0], "\ndocs:\n", docs[0])
print("\nlabel: ", labels[1000], "\ndocs:\n", docs[1000])
