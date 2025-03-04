word_count = {}

dat = open("song.txt")
for line in dat:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word not in word_count:
            word_count[word] = 1
            continue
        word_count[word] = word_count[word] + 1
dat.close()
unique_words = 0
for word in word_count:
    if word_count[word] == 1:
        unique_words += 1
    print(f"{word} : {word_count[word]}")
print(unique_words)