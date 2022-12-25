import random


# 将数据集转化为bio的形式
def convert_dataset2bio(
    in_folder = "ccks2021/raw", 
    out_folder = "ccks2021/bio",
):

    def read_raw_file(data_path, id):
        data_x = []
        data_y = []
        data_id = []

        # 训练集和验证集
        if id == 0 or id == 1:
            with open(data_path, "r", encoding = "utf-8") as f:
                data = f.readlines()
                data = [j.strip() for j in data]
                if data[-1] != "":
                    data.append("")
                
                tmp_x = ""
                tmp_y = []
                for i in data:
                    if i == "":
                        data_x.append(tmp_x)
                        data_y.append(tmp_y)
                        tmp_x = ""
                        tmp_y = []
                    else:
                        tmp_x = tmp_x + i.split(" ")[0]
                        tmp_y.append(i.split(" ")[1])        
                return [data_x, data_y, list(range(len(data_x)))]
        # 测试集
        else:
            with open(data_path, "r", encoding = "utf-8") as f:
                data = f.readlines()
                data = [j.strip() for j in data]
                data = [i for i in data if i != ""]
                for i in data:
                    data_id.append(i.split("\u0001")[0])
                    data_x.append(i.split("\u0001")[1])
                    data_y.append(["O"] * len(i.split("\u0001")[1]))
                return [data_x, data_y, data_id]


    def convert_test(input_path, output_path):
        with open(input_path, "r", encoding = "utf-8") as f:
            data = f.readlines()
            data = [i.strip() for i in data]
            data = [i for i in data if i != ""]
            data = [i.split("\u0001")[1] for i in data]
            
            with open(output_path, "w", encoding = "utf-8") as f:
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        f.write(f"{data[i][j]}\tO\n")
                    f.write("\n")


    def convert_bio(input_path, output_path):
        data_x = []
        data_y = []
        data_id = []

        with open(input_path, "r", encoding = "utf-8") as f:
            data = f.readlines()
            data = [j.strip() for j in data]
            if data[-1] != "":
                data.append("")
            
            tmp_x = ""
            tmp_y = []
            for i in data:
                if i == "":
                    data_x.append(tmp_x)
                    data_y.append(tmp_y)
                    tmp_x = ""
                    tmp_y = []
                else:
                    tmp_x = tmp_x + i.split(" ")[0]
                    tmp_y.append(i.split(" ")[1])

            with open(output_path, "w", encoding = "utf-8") as f:
                for i in range(len(data_x)):
                    for j in range(len(data_x[i])):
                        label = data_y[i][j]
                        if label.split("-")[0] in ("E", "S"):
                            label = "I-" + label.split("-")[1]
                        f.write(f"{data_x[i][j]}\t{label}\n")
                    f.write("\n")

    train_input = os.path.join(in_folder, "train.txt")
    train_output = os.path.join(out_folder, "train.txt")
    valid_input = os.path.join(in_folder, "dev.txt")
    valid_output = os.path.join(out_folder, "dev.txt")
    test_input = os.path.join(in_folder, "test.txt")
    test_output = os.path.join(out_folder, "test.txt")

    convert_bio(train_input, train_output)
    convert_bio(valid_input, valid_output)
    convert_test(test_input, test_output)


# 检查生成的result.txt中的label长度和原文一致
def check_txt_valid(
    raw_path = "ccks2021/raw/test.txt",
    result_path = "checkpoints/bert-base/token_labels_.txt",
):
    with open(raw_path, "r", encoding = "utf-8") as f:
        data = f.readlines()
        data = [i.strip() for i in data]
        data = [i for i in data if i != ""]

        ids1 = []
        sentences1 = []
        for i in data:
            ids1.append(i.split("\u0001")[0])
            sentences1.append(i.split("\u0001")[1])

    with open(result_path, "r", encoding = "utf-8") as f:
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
        if len(sentences1[i]) != len(labels2[i].split(" ")):
            print(i, len(sentences1[i]), len(labels2[i].split(" ")))


# !!!NOTE: results输入时一定要保证得分高的txt排在前面
def ensemble_by_vote(
    results,
    out_file = "results.txt",
):
    assert isinstance(results, list)

    labels = []
    for i, f in enumerate(results):
        with open(f, "r", encoding = "utf-8") as f:
            data = f.readlines()
            data = [i.strip() for i in data]
            data = [i for i in data if i != ""]

            if i == 0:
                ids = [i.split("\u0001")[0] for i in data]
                sentences = [i.split("\u0001")[1] for i in data]
            label = [i.split("\u0001")[2] for i in data]
            label = [i.split(" ") for i in label]
        labels.append(label)

    new_labels = []
    # 每句话
    for i in range(len(ids)):
        new_label = ""
        # 每个字词
        for j in range(len(labels[0][i])):
            # 每种答案
            answer = {}
            max_times = 0
            for k in range(len(labels)):
                if labels[k][i][j] not in answer:
                    answer[labels[k][i][j]] = [k]
                else:
                    answer[labels[k][i][j]].append(k)
                max_times = max(max_times, len(answer[labels[k][i][j]]))
            
            min_id = len(results)
            s = None
            for a in answer:
                if len(answer[a]) == max_times:
                    if min(answer[a]) < min_id:
                        s = a
            new_label = new_label + s + " "
        new_labels.append(new_label[: -1])
    
    with open(out_file, "w", encoding = "utf-8") as f:
        for i in range(len(ids)):
            f.write(f"{ids[i]}\u0001{sentences[i]}\u0001{new_labels[i]}\n")


if __name__ == "__main__":
    # top7
    ensemble_by_vote([
        "checkpoints/chinese-electra-180g-base-discriminator/token_labels_.txt",
        "checkpoints/chinese-lert-base/token_labels_.txt",
        "checkpoints/chinese-pert-base/token_labels_.txt",
        "checkpoints/bert-base-chinese/token_labels_.txt", 
        "checkpoints/chinese-bert-wwm-ext/token_labels_.txt",
        "checkpoints/chinese-roberta-wwm-ext/token_labels_.txt",
        "checkpoints/chinese-macbert-base/token_labels_.txt",
    ], out_file = "checkpoints/top7.txt")
    check_txt_valid(result_path = "checkpoints/top7.txt")

    # top6
    ensemble_by_vote([
        "checkpoints/chinese-electra-180g-base-discriminator/token_labels_.txt",
        "checkpoints/chinese-lert-base/token_labels_.txt",
        "checkpoints/chinese-pert-base/token_labels_.txt",
        "checkpoints/bert-base-chinese/token_labels_.txt", 
        "checkpoints/chinese-bert-wwm-ext/token_labels_.txt",
        "checkpoints/chinese-roberta-wwm-ext/token_labels_.txt",
    ], out_file = "checkpoints/top6.txt")
    check_txt_valid(result_path = "checkpoints/top6.txt")

    # top5
    ensemble_by_vote([
        "checkpoints/chinese-electra-180g-base-discriminator/token_labels_.txt",
        "checkpoints/chinese-lert-base/token_labels_.txt",
        "checkpoints/chinese-pert-base/token_labels_.txt",
        "checkpoints/bert-base-chinese/token_labels_.txt", 
        "checkpoints/chinese-bert-wwm-ext/token_labels_.txt",
    ], out_file = "checkpoints/top5.txt")
    check_txt_valid(result_path = "checkpoints/top5.txt")
