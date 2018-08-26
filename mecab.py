# 形態素解析をしてください
import MeCab

mecab = MeCab.Tagger("-Ochasen")
print(mecab.parse("ダックスフンドが歩いている。"))
