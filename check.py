import os


if __name__ == "__main__":

    with open("ccks2021/raw/test.txt", "r", encoding = "utf-8") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
        data = [i for i in data if i != ""]

        ids1 = []
        sentences1 = []
        for i in data:
            ids1.append(i.split("\u0001")[0])
            sentences1.append(i.split("\u0001")[1])


    with open("checkpoints/bert-base/token_labels_.txt", "r", encoding = "utf-8") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
        data = [i for i in data if i != ""]

        ids2 = []
        sentences2 = []
        labels2 = []
        for i in data:
            ids2.append(i.split("\u0001")[0])
            sentences2.append(i.split("\u0001")[1])
            labels2.append(i.split("\u0001")[2])

    assert ids1 == ids2
    for i in range(len(sentences1)):
        if sentences1[i] != sentences2[i]:
            print(i, len(sentences1[i]), len(sentences2[i]), sentences1[i], sentences2[i])