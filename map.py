import torch
import matplotlib.pyplot as plt
import sys
import os

def save_attention_map(pth_path, save_path, source_tokens, target_tokens):
    
    attention_map = torch.load(pth_path, map_location='cpu')

    print(f"Attention Map Shape: {attention_map.shape}")

    plt.figure(figsize=(len(source_tokens), len(target_tokens)))

    plt.imshow(attention_map, cmap='gray_r', aspect='auto')

    plt.xticks(range(len(source_tokens)), source_tokens, fontweight='bold', rotation=30)
    plt.yticks(range(len(target_tokens)), target_tokens, fontweight='bold', rotation=30)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"saved: {save_path}")


test="test.2016"
checkpoint = "checkpoints/multi30k-en2de/vit_base_patch16_384/vit_base_patch16_384-mask0"
source = "data/multi30k-en-de/"+test+".en"
target = checkpoint+"/map/hypo.sorted"
map_path = checkpoint+"/visualization/"
hypo = checkpoint+"/map/hypo.txt"

source_tokens = [line.strip().split()+['<eos>']+['<img>'] for line in open(source, 'r', encoding='utf-8')]
target_tokens = [line.strip().split()+['<eos>'] for line in open(target, 'r', encoding='utf-8')]
order = [line.strip().split()[0] for line in open(hypo, 'r', encoding='utf-8')]

os.makedirs(checkpoint+'/map/'+test, exist_ok=True)

#n=sys.argv[1]

for n in range(len(target_tokens)):

    input_pth = map_path+str(order.index(str(n)))+'map.pth'
    output_img = checkpoint+'/map/'+test+'/'+str(n)+'map.png'
    save_attention_map(input_pth, output_img, source_tokens[int(n)], target_tokens[int(n)])
