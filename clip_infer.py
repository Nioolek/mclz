import torch
from PIL import Image
import open_clip
from pathlib import Path
import cv2
import numpy as np
from itertools import product

'''
BIT 1 : 抽烟
BIT 2 : 赤膊
BIT 3 : 老鼠
BIT 4 : 猫
BIT 5 : 狗

[1, 512] @ [512, 2]
'''
device = torch.device('cuda:0')
m = 'ViT-B-32'
# model, _, preprocess = open_clip.create_model_and_transforms(m, pretrained='datacomp_xl_s13b_b90k', device=device)
model, _, preprocess = open_clip.create_model_and_transforms(m, device=device)
tokenizer = open_clip.get_tokenizer(m)

base_token = ['a kitchen with no smokers, no shirtless people, no rats, no cats, no dogs']
tokens = ['an image of a smoker',
          'an image of a shirtless person',
          'an image with a mouse',
          'an image with a cat',
          'an image with a dog']

text_features = []

for i, tk in enumerate(product(base_token, tokens)):
    text = tokenizer(tk)
    text = text.to(device)
    tf = model.encode_text(text)
    tf /= tf.norm(dim=-1, keepdim=True)
    tf = tf.T
    tf = tf.contiguous()
    text_features.append(tf)

all_text_features = torch.stack(text_features)

root = Path('初赛试题')

results = []
with torch.no_grad(), torch.cuda.amp.autocast():
    for d in root.iterdir():
        for f in d.iterdir():
            labels = []
            if f.suffix == '.jpg':
                pil = Image.open(f)
                image = preprocess(pil).unsqueeze(0)
                image = image.to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.unsqueeze(0)
                prob_features = 100.0 * image_features @ all_text_features
                text_probs = prob_features.softmax(dim=-1)
                text_probs = text_probs.squeeze(1)
                s = ''.join(map(str, text_probs.argmax(-1).tolist()))
                labels.append(int(s))
            if f.suffix == '.ts':
                _labels = []
                cap = cv2.VideoCapture(str(f))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame)
                    image = preprocess(pil).unsqueeze(0)
                    image = image.to(device)
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.unsqueeze(0)
                    prob_features = 100.0 * image_features @ all_text_features
                    text_probs = prob_features.softmax(dim=-1)
                    text_probs = text_probs.squeeze(1)
                    s = ''.join(map(str, text_probs.argmax(-1).tolist()))
                    _labels.append(int(s))
                _labels, counts = np.unique(_labels, return_counts=True)
                labels.append(_labels[counts.argmax()])
            if not labels:
                continue
            assert len(labels) == 1
            results.append(
                dict(
                    filename=f.name,
                    labels=labels[0],
                )
            )
print(results)
