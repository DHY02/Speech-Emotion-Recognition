from pathlib import Path
import os
import utils
import shutil
import models
from predict import predict
"""
    对同一audio_name的5个采样音频进行情感预测，并制作dpo数据集
"""
emo_label = "angry"

def make_dpo_corpus(config, model):
    # 对同一audio_name的5个采样音频进行推理，将最小acc的和最高acc的采样分别作为拒绝样本和接受样本记录下来
    samples_dir = "/home/CosyVoice/examples/libritts/cosyvoice2/exp/cosyvoice/casia_train_angry50_inf/sft"
    ori_dir = "/home/CosyVoice/examples/libritts/cosyvoice2/data/casia_train"

    worst_acc = {}
    worst_acc_path = {}
    best_acc = {}
    best_acc_path = {}

    acc_total = 0
    cnt = 0

    for entry in os.listdir(samples_dir):
        samp_dir_path = os.path.join(samples_dir, entry)
        if os.path.isdir(samp_dir_path) and "samp" in entry:
            for wav in os.listdir(samp_dir_path):
                if ".wav" in wav:
                    audio_name = wav.split('.')[0].strip()
                    
                    samp_wav_path = os.path.join(samp_dir_path, wav)

                    txt_name = audio_name[:-2] + ".normalized.txt"
                    text_path = os.path.join(ori_dir, txt_name)

                    _, result_prob = predict(config, audio_path=samp_wav_path, model=model)
                    # angry index = 0
                    acc = result_prob[0]
                    cnt += 1
                    acc_total += acc
                    
                    if audio_name not in worst_acc or worst_acc[audio_name] > acc:
                        worst_acc[audio_name] = acc
                        worst_acc_path[audio_name] = (samp_wav_path, text_path)
                    if audio_name not in best_acc or best_acc[audio_name] < acc:
                        best_acc[audio_name] = acc
                        best_acc_path[audio_name] = (samp_wav_path, text_path)
    
    avg_acc = acc_total / cnt
    print(f"***根据{cnt}个样本的统计，{emo_label} 情感的平均acc为{avg_acc}")
    script_path = Path(".")
    cosyvoice_dir_path = script_path / "cosyvoice"
    os.makedirs(cosyvoice_dir_path, exist_ok=True)
    with open(cosyvoice_dir_path / "accuracy_beforeDPO.txt", "a", encoding="utf-8") as f:
        f.write(f"{emo_label} acc: {avg_acc}\n")

    reject_label_dir = cosyvoice_dir_path / f"reject_{emo_label}"
    receive_label_dir = cosyvoice_dir_path / f"receive_{emo_label}"
    os.makedirs(reject_label_dir, exist_ok=True)
    os.makedirs(receive_label_dir, exist_ok=True)
    for au_name, (wav_path, text_path) in worst_acc_path.items():
        wav_name = au_name[:-2] + ".wav"
        shutil.copy(wav_path, reject_label_dir / wav_name)
        shutil.copy(text_path, reject_label_dir)

    for au_name, (wav_path, text_path) in best_acc_path.items():
        wav_name = au_name[:-2] + ".wav"
        shutil.copy(wav_path, receive_label_dir / wav_name)
        shutil.copy(text_path, receive_label_dir)
    
    print("done!")


if __name__ == '__main__':
    config = utils.parse_opt()
    model = models.load(config)
    make_dpo_corpus(config, model)
