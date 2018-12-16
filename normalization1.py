# neologdnをインポートしてください
import neologdn

# 半角カタカナを全角に統一
a = neologdn.normalize("ｶﾀｶﾅ")
print(a)

# 長音短縮
b = neologdn.normalize("長音短縮ウェーーーーイ")
print(b)

# 似た文字の統一
c = neologdn.normalize("いろんなハイフン˗֊‐‑‒–⁃⁻₋−")
print(c)

# 全角英数字を半角に統一 + 不要なスペースの削除
d = neologdn.normalize("　　　ＤＬ　　デ  ィ ープ ラ  ーニング　　　　　")
print(d)

# 繰り返しの制限
e = neologdn.normalize("かわいいいいいいいいい", repeat=6)
print(e)
