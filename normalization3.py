import re
def normalize_number(text):
    # 以下に回答を作成してください
    replaced_text = re.sub("\d", "!", text)
    return replaced_text

replaced_text = normalize_number("終日は前日よりも39.03ドル(0.19%)高い。")
print(replaced_text)
