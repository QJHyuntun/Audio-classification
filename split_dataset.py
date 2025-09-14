import os
import shutil
import random

# ===============================
# 参数设置
# ===============================
SRC_DIR = "data/wav_dataset_raw"   # 原始数据目录，每个类别一个子文件夹
DST_DIR = "data/wav_dataset_split" # 输出目录
SPLIT_RATIO = [0.7, 0.15, 0.15]    # train/val/test 比例
SEED = 42                          # 随机种子，保证可复现

label_map = {
    0: "正常语音",
    1: "声带结节",
    2: "声带息肉",
    3: "声带麻痹",
    4: "声带水肿"
}


def make_dirs():
    """创建 train/val/test 目录结构"""
    for split in ["train", "val", "test"]:
        for label in label_map.keys():
            out_dir = os.path.join(DST_DIR, split, str(label))
            os.makedirs(out_dir, exist_ok=True)

def split_dataset():
    random.seed(SEED)
    make_dirs()

    for label, name in label_map.items():
        class_dir = os.path.join(SRC_DIR, str(label))
        if not os.path.exists(class_dir):
            print(f"[警告] 类别 {label}-{name} 文件夹不存在，跳过。")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * SPLIT_RATIO[0])
        n_val = int(n_total * SPLIT_RATIO[1])
        n_test = n_total - n_train - n_val

        print(f"类别 {label}-{name}: 总数 {n_total} -> train {n_train}, val {n_val}, test {n_test}")

        # 分配
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        # 拷贝文件
        for f in train_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(DST_DIR, "train", str(label), f))
        for f in val_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(DST_DIR, "val", str(label), f))
        for f in test_files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(DST_DIR, "test", str(label), f))

    print("✅ 数据集划分完成！输出目录:", DST_DIR)


if __name__ == "__main__":
    split_dataset()
