#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import open_clip
from open_clip import tokenizer
from deep_translator import GoogleTranslator
from collections import defaultdict
import os
import shutil
import random
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import numpy as np
random.seed(42)
np.random.seed(42)


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[3]:


# RN50x64
high_model, _, high_preprocess = open_clip.create_model_and_transforms('RN50x64', pretrained='openai')
high_model.eval()
high_model = high_model.to(device)


# In[4]:


# RN50
low_model, _, low_preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
low_model.eval()
low_model = low_model.to(device)


# In[5]:


# DeeplTranslatorのインスタンスを作成
translator = GoogleTranslator(source='ja', target='en')

# 辞書の定義(現状ベスト)
descriptions = {
    "NO_A": "上空から緑色や黃緑色の草が生い茂った箇所を見下ろした画像で全体をくまなく探してもゴミなどの人工物、白色の小さい物体、黒色の物体、木の枝は確実に写っていない",
    "NO_B": "上空から緑色や黃緑色の草が砂浜の上から生えているのを見下ろした画像で全体をくまなく探してもゴミなどの人工物、白色の小さい物体、黒色の物体、木の枝は確実に写っていない",
    "NO_C": "上空から緑色の葉が生えた木と隙間に砂浜が見られる箇所で画像で全体をくまなく探してもゴミなどの人工物、白色の小さい物体、黒色の物体、木の枝は確実に写っていない",
    "NO_D":"上空からクリーム色の砂浜の上に少し草が生えている箇所を見下ろした画像で全体をくまなく探してもゴミなどの人工物、白色の小さい物体、黒色の物体、木の枝は確実に写っていない",
    "NO_E":"上空からクリーム色の砂浜を見下ろした画像で全体をくまなく探してもゴミなどの人工物、白色の小さい物体、黒色の物体、木の枝は確実に写っていない",
    "YES_A":"上空から緑色や黃緑色の草が生い茂った箇所を見下ろした画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある",
    "YES_B":"上空から緑色の葉が生えた木を見下ろした画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある",
    "YES_C":"上空から緑色や黃緑色の草が砂浜の上から生えているのを見下ろした画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある",
    "YES_D":"上空から緑色の葉が生えた木と隙間に砂浜が見られる箇所で画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある",
    "YES_E":"上空からクリーム色の砂浜の上に少し草が生えている箇所を見下ろした画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある",
    "YES_F":"上空からクリーム色の砂浜を見下ろした画像で全体をくまなく探した結果ゴミなどの人工物が写っている可能性がある"
}
descriptions = {translator.translate(key): translator.translate(value) for key, value in descriptions.items()}

# 翻訳した辞書を表示
print(descriptions)


# In[ ]:


# 元のフォルダパス
source_base_path = "imgs/update_img"
# コピー先のフォルダパス
destination_path = "imgs/clip_test"

# コピー先のフォルダを作成
os.makedirs(destination_path, exist_ok=True)

# サブフォルダを取得
subfolders = [f for f in os.listdir(source_base_path) if os.path.isdir(os.path.join(source_base_path, f))]

# 各サブフォルダからランダムにn枚ずつ画像をコピー
for subfolder in subfolders:
    subfolder_path = os.path.join(source_base_path, subfolder)
    images = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
    
    # ランダムに2枚選択
    selected_images = random.sample(images, 10)
    
    for image in selected_images:
        source_file = os.path.join(subfolder_path, image)
        destination_file = os.path.join(destination_path, f"{subfolder}_{image}")
        
        # 画像をコピー
        shutil.copy(source_file, destination_file)


# In[ ]:


clip_test_path = destination_path
result_path = "imgs/clip_test_result"
# 結果フォルダを作成
os.makedirs(result_path, exist_ok=True)

# 画像の予測とプロット
for image_name in os.listdir(clip_test_path):
    image_path = os.path.join(clip_test_path, image_name)
    if os.path.isfile(image_path):
        # 画像の読み込みと前処理
        image = Image.open(image_path).convert("RGB")
        high_input = high_preprocess(image).unsqueeze(0).to(device)
        low_input = low_preprocess(image).unsqueeze(0).to(device)

        # テキストの前処理
        texts = list(descriptions.values())
        text_tokens = open_clip.tokenize(texts).to(device)

        # 予測
        with torch.no_grad():
            high_img_embedding = high_model.encode_image(high_input)
            high_text_embedding = high_model.encode_text(text_tokens)
            low_img_embedding = low_model.encode_image(low_input)
            low_text_embedding = low_model.encode_text(text_tokens)

        # コサイン類似度を計算
        high_probs = (100 * high_img_embedding @ high_text_embedding.T).softmax(dim=-1)
        low_probs = (100 * low_img_embedding @ low_text_embedding.T).softmax(dim=-1)

        # 結果のプロット
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # 画像の表示
        ax[0].imshow(image)
        ax[0].axis('off')
        ax[0].set_title(f"Image: {image_name}")

        # 予測結果の棒グラフ
        labels = list(descriptions.keys())
        high_values = high_probs.squeeze().detach().cpu().numpy()
        low_values = low_probs.squeeze().detach().cpu().numpy()
        x = np.arange(len(labels))
        width = 0.35
        ax[1].bar(x - width/2, high_values, width, label='High Model', color='blue', alpha=0.5)
        ax[1].bar(x + width/2, low_values, width, label='Low Model', color='green', alpha=0.5)
        ax[1].set_title("Model Predictions")
        ax[1].set_ylabel("Probability (%)")
        ax[1].legend()
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=45, ha='right')

        # 結果の保存
        result_image_path = os.path.join(result_path, f"{image_name}_result.png")
        plt.savefig(result_image_path)
        plt.close(fig)

print("予測結果の保存が完了しました。")


# In[8]:


# 画像フォルダのパス
source_base_path = "imgs/update_img"
result_base_path = "imgs/clip_result"

# 結果フォルダを作成
os.makedirs(result_base_path, exist_ok=True)
high_result_path = os.path.join(result_base_path, "high")
low_result_path = os.path.join(result_base_path, "low")
os.makedirs(high_result_path, exist_ok=True)
os.makedirs(low_result_path, exist_ok=True)

# サブフォルダを取得
subfolders = [f for f in os.listdir(source_base_path) if os.path.isdir(os.path.join(source_base_path, f))]


# 画像の予測と分類
for subfolder in subfolders:
    subfolder_path = os.path.join(source_base_path, subfolder)
    high_yes_path = os.path.join(high_result_path, subfolder, "yes")
    high_no_path = os.path.join(high_result_path, subfolder, "no")
    low_yes_path = os.path.join(low_result_path, subfolder, "yes")
    low_no_path = os.path.join(low_result_path, subfolder, "no")
    os.makedirs(high_yes_path, exist_ok=True)
    os.makedirs(high_no_path, exist_ok=True)
    os.makedirs(low_yes_path, exist_ok=True)
    os.makedirs(low_no_path, exist_ok=True)
    
    for image_name in os.listdir(subfolder_path):
        image_path = os.path.join(subfolder_path, image_name)
        if os.path.isfile(image_path):

            # 画像の読み込みと前処理
            image = Image.open(image_path).convert("RGB")
            high_input = high_preprocess(image).unsqueeze(0).to(device)
            low_input = low_preprocess(image).unsqueeze(0).to(device)

            # テキストの前処理
            texts = list(descriptions.values())
            text_tokens = open_clip.tokenize(texts).to(device)

            # 予測
            with torch.no_grad():
                high_img_embedding = high_model.encode_image(high_input)
                high_text_embedding = high_model.encode_text(text_tokens)
                low_img_embedding = low_model.encode_image(low_input)
                low_text_embedding = low_model.encode_text(text_tokens)

            # コサイン類似度を計算
            high_probs = (100 * high_img_embedding @ high_text_embedding.T).softmax(dim=-1)
            low_probs = (100 * low_img_embedding @ low_text_embedding.T).softmax(dim=-1)

            # 物体があるかないかの判断
            high_max_prob, high_max_index = torch.max(high_probs, dim=-1)
            low_max_prob, low_max_index = torch.max(low_probs, dim=-1)
            high_label = list(descriptions.keys())[high_max_index]
            low_label = list(descriptions.keys())[low_max_index]

            # high_modelの分類
            if high_label.startswith("YES"):
                shutil.copy(image_path, os.path.join(high_yes_path, image_name))
            else:
                shutil.copy(image_path, os.path.join(high_no_path, image_name))


            # low_modelの分類
            if low_label.startswith("YES"):
                shutil.copy(image_path, os.path.join(low_yes_path, image_name))
            else:
                shutil.copy(image_path, os.path.join(low_no_path, image_name))

print("画像の分類と保存が完了しました。")


# In[9]:


# コピー元のフォルダパス
source_path = result_base_path

# コピー先のフォルダパス
destination_path = result_base_path+"_remove"

# フォルダのコピー
shutil.copytree(source_path, destination_path)


for root, dirs, files in os.walk(destination_path):
    for dir_name in dirs:
        sub_dir_path = os.path.join(root, dir_name)
        for sub_root, sub_dirs, sub_files in os.walk(sub_dir_path):
            for _ in sub_dirs:
                new_dir_path = os.path.join(sub_root, "yesなのにno")
                os.makedirs(new_dir_path, exist_ok=True)

# 不要なディレクトリを削除
dirs_to_remove = [
    os.path.join(destination_path, "low", "yesなのにno"),
    os.path.join(destination_path, "high", "yesなのにno")
]

for dir_path in dirs_to_remove:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"{dir_path} を削除しました。")


# In[ ]:


import os
import shutil

# ベースパスの設定
update_base_path = "imgs/update_img_remove"
clip_base_path = "imgs/clip_result_remove"

# update_img_removeのサブフォルダを取得
update_subfolders = [f for f in os.listdir(update_base_path) if os.path.isdir(os.path.join(update_base_path, f))]

# clip_result_removeのサブフォルダを取得
clip_subfolders = ["high", "low"]

for clip_subfolder in clip_subfolders:
    for update_subfolder in update_subfolders:
        object_folder = os.path.join(update_base_path, f"{update_subfolder}_object")
        no_folder = os.path.join(clip_base_path, clip_subfolder, update_subfolder, "no")
        yes_nano_folder = os.path.join(clip_base_path, clip_subfolder, update_subfolder, "yesなのにno")

        # フォルダが存在するか確認
        if os.path.exists(object_folder) and os.path.exists(no_folder) and os.path.exists(yes_nano_folder):
            # objectフォルダ内の画像ファイル名を取得
            object_images = set(os.listdir(object_folder))
            total_object_images = len(object_images)

            moved_images = 0
            image=0
            # noフォルダ内の画像をチェックして移動
            for image_name in os.listdir(no_folder):
                image += 1
                if image_name in object_images:
                    source_image_path = os.path.join(no_folder, image_name)
                    destination_image_path = os.path.join(yes_nano_folder, image_name)
                    shutil.move(source_image_path, destination_image_path)
                    moved_images += 1

            # 正答率の計算
            if total_object_images > 0:
                accuracy = ((total_object_images - moved_images) / total_object_images) * 100
            else:
                accuracy = 0
            
            score= moved_images/image*100

            print(f"もし移動しなかったらnoに入っている物体の画像の割合は{score:.2f}%")
            print(f"{yes_nano_folder} に移動した画像の枚数: {moved_images}")
            print(f"{object_folder} の総参照画像枚数: {total_object_images}")
            print(f"{object_folder} の正答率: {accuracy:.2f}%")

print("画像の移動が完了しました。")


# In[31]:


# コピー元のフォルダパス
source_base_paths = ["imgs/clip_result_remove", "imgs/clip_result"]

# 任意の画像の枚数
num_images = 6048

# 画像のコピーと反転処理
for source_base_path in source_base_paths:
    if "remove" in source_base_path:
        destination_base_path = "imgs/update_train_imgs_remove/train_img_6048"
    else:
        destination_base_path = "imgs/update_train_imgs_default/train_img_6048"
    
    # コピー先のフォルダを作成
    os.makedirs(destination_base_path, exist_ok=True)
    
    high_path = os.path.join(source_base_path, "high")
    # サブフォルダを取得
    for subfolder in os.listdir(high_path):
        no_folder_path = os.path.join(high_path, subfolder, "no")
        if os.path.exists(no_folder_path):
            images = [f for f in os.listdir(no_folder_path) if os.path.isfile(os.path.join(no_folder_path, f))]
            
            # 画像が足りない場合は反転処理と回転処理
            while len(images) < num_images:
                image_name = random.choice(images)
                image_path = os.path.join(no_folder_path, image_name)
                image = Image.open(image_path)
                
                # ランダムに反転処理
                if random.choice([True, False]):
                    image = ImageOps.mirror(image)
                if random.choice([True, False]):
                    image = ImageOps.flip(image)
                
                # ランダムに回転処理
                image = image.rotate(random.choice([0, 90, 180, 270]))
                
                # 新しい画像名を生成
                new_image_name = f"aug_{len(images)}_{image_name}"
                new_image_path = os.path.join(no_folder_path, new_image_name)
                image.save(new_image_path)
                images.append(new_image_name)
            
            # 画像をランダムに選択してコピー
            selected_images = random.sample(images, num_images)
            for image_name in selected_images:
                image_path = os.path.join(no_folder_path, image_name)
                if "remove" in source_base_path:
                    dest_folder_path = os.path.join(destination_base_path, "6048", "r" + subfolder)
                else:
                    dest_folder_path = os.path.join(destination_base_path, "6048","d" + subfolder)
                os.makedirs(dest_folder_path, exist_ok=True)
                shutil.copy(image_path, dest_folder_path)

print("画像のコピーと反転処理が完了しました。")

