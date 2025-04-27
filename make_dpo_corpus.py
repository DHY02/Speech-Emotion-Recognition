from pathlib import Path
import os
import utils
import shutil
import models
from predict import predict
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
"""
    对同一audio_name的5个采样音频进行情感预测，并制作dpo数据集
"""
emo_acc = {}
acc_label = "beforeDPO"

# 验证情感分布
def print_distribution(names, title):
    unique, counts = np.unique([name[:-2].split('_')[-1] for name in names], return_counts=True)
    print(f"{title}情感分布: {dict(zip(unique, counts))}")

def make_dpo_corpus(config, model):
    # 对同一audio_name的5个采样音频进行推理，将最小acc的和最高acc的采样分别作为拒绝样本和接受样本记录下来
    samples_dir = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/exp/cosyvoice/casia_dpo/sft"
    ori_dir = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia"

    worst_acc = {}
    worst_acc_path = {}
    best_acc = {}
    best_acc_path = {}

    emo_acc_total = {}
    emo_cnt = {}

    for entry in os.listdir(samples_dir):
        samp_dir_path = os.path.join(samples_dir, entry)
        if os.path.isdir(samp_dir_path) and "samp" in entry:
            for wav in os.listdir(samp_dir_path):
                if ".wav" in wav:
                    audio_name = wav.split('.')[0].strip()
                    emotion_name = audio_name.split('_')[-1].strip()
                    samp_wav_path = os.path.join(samp_dir_path, wav)

                    txt_name = audio_name[:-2] + ".normalized.txt"
                    text_path = os.path.join(ori_dir, txt_name)

                    _, result_prob = predict(config, audio_path=samp_wav_path, model=model)
                    # angry index = 0
                    acc = result_prob[0]
                    if emotion_name not in emo_cnt:
                        emo_cnt[emotion_name] = 0
                    else:
                        emo_cnt[emotion_name] += 1
                    if emotion_name not in emo_acc_total:
                        emo_acc_total[emotion_name] = 0
                    else:
                        emo_acc_total[emotion_name] += acc
                    
                    if audio_name not in worst_acc or worst_acc[audio_name] > acc:
                        worst_acc[audio_name] = acc
                        worst_acc_path[audio_name] = (samp_wav_path, text_path)
                    if audio_name not in best_acc or best_acc[audio_name] < acc:
                        best_acc[audio_name] = acc
                        best_acc_path[audio_name] = (samp_wav_path, text_path)
    
    
    script_path = Path(".")
    cosyvoice_dir_path = script_path / "cosyvoice"
    os.makedirs(cosyvoice_dir_path, exist_ok=True)
    for k, v in emo_acc_total.items():
        avg_acc = v / emo_cnt[k]
        print(f"***根据{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
        with open(cosyvoice_dir_path / f"accuracy_{acc_label}.txt", "a", encoding="utf-8") as f:
            f.write(f"{k} acc: {avg_acc}\n")

    reject_label_dir = cosyvoice_dir_path / f"reject_dpo"
    receive_label_dir = cosyvoice_dir_path / f"receive_dpo"
    os.makedirs(reject_label_dir, exist_ok=True)
    os.makedirs(receive_label_dir, exist_ok=True)

    # 将数据集分为训练集、验证集、测试集，根据情感label进行分层采样
    audio_names = np.array(list(worst_acc_path.keys()))
    emotion_labels = np.array([name[:-2].split('_')[-1] for name in audio_names])
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, temp_idx in split1.split(audio_names, emotion_labels):
        train_names = audio_names[train_idx]  # 80%训练集
        temp_names = audio_names[temp_idx]    # 20%临时集
        temp_emotions = emotion_labels[temp_idx]
    
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in split2.split(temp_names, temp_emotions):
        val_names = temp_names[val_idx]   # 10%验证集
        test_names = temp_names[test_idx] # 10%测试集

 
    print_distribution(train_names, "训练集")
    print_distribution(val_names, "验证集")
    print_distribution(test_names, "测试集")
    
    # 重组为字典
    train_data = {name: worst_acc_path[name] for name in train_names}
    val_data = {name: worst_acc_path[name] for name in val_names}
    test_data = {name: worst_acc_path[name] for name in test_names}

    # {train,valid,test}/receive_dpo
    for au_name, (wav_path, text_path) in worst_acc_path.items():
        wav_name = au_name[:-2] + ".wav"
        shutil.copy(wav_path, reject_label_dir / wav_name)
        shutil.copy(text_path, reject_label_dir)

    # {train,valid,test}/reject_dpo
    for au_name, (wav_path, text_path) in best_acc_path.items():
        wav_name = au_name[:-2] + ".wav"
        shutil.copy(wav_path, receive_label_dir / wav_name)
        shutil.copy(text_path, receive_label_dir)
    


    print("done!")


if __name__ == '__main__':
    config = utils.parse_opt()
    model = models.load(config)
    make_dpo_corpus(config, model)
