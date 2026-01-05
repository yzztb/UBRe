import matplotlib.pyplot as plt
import numpy as np


def plot_pic_2(x_lst, y_lst0, y_lst1, y_lst2, y_lst3,y_lst0_acc,y_lst1_acc,y_lst2_acc,y_lst3_acc, save_path):
    fontsize = 7
    linewidth = 1
    markersize = 4
    plt.figure(figsize=(5.5, 3.1))
    plt.plot(x_lst, y_lst0, marker = 'o', color='#92A6BE', linewidth=linewidth, markersize=markersize,linestyle='--', label='mistral-ASR')
    plt.plot(x_lst, y_lst1, marker = 'o', color='#008000', linewidth=linewidth, markersize=markersize,linestyle='--', label='Qwen-ASR')
    plt.plot(x_lst, y_lst2, marker = 'o', color='#D2B48C', linewidth=linewidth, markersize=markersize,linestyle='--', label='llama-ASR')
    plt.plot(x_lst, y_lst3, marker = 'o', color='#FFA500', linewidth=linewidth, markersize=markersize,linestyle='--', label='deepseek-ASR')

    plt.plot(x_lst, y_lst0_acc, marker = '^', color='#92A6BE', linewidth=linewidth, markersize=markersize, label='mistral-BA')
    plt.plot(x_lst, y_lst1_acc, marker = '^', color='#008000', linewidth=linewidth, markersize=markersize, label='Qwen-BA')
    plt.plot(x_lst, y_lst2_acc, marker = '^', color='#D2B48C', linewidth=linewidth, markersize=markersize, label='llama-BA')
    plt.plot(x_lst, y_lst3_acc, marker = '^', color='#FFA500', linewidth=linewidth, markersize=markersize, label='deepseek-BA')


    # plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Similarity threshold', fontsize=fontsize)
    plt.ylabel('Values', fontsize=fontsize)

    plt.grid(True, linestyle='--', alpha=0.7) 
    plt.legend(ncol=4,fontsize=fontsize,loc='upper right')

    plt.ylim(top=1)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(save_path, dpi=300)

x_lst = [0.8, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]

deepseek = [0.7902, 0.817, 0.83, 0.85, 0.865, 0.87, 0.873, 0.876 ]
llama = [0.816, 0.838, 0.835, 0.863, 0.87, 0.88, 0.87, 0.875]
qwen = [0.805, 0.812, 0.83, 0.859, 0.857, 0.879, 0.882, 0.883]
mistral = [0.813, 0.823, 0.85, 0.86, 0.878, 0.883, 0.878, 0.88]

deepseek_acc = [0.92, 0.934, 0.94, 0.942, 0.96, 0.96, 0.956, 0.961]
llama_acc = [0.926, 0.933, 0.917, 0.935, 0.951, 0.96, 0.97,  0.962]
qwen_acc = [0.92, 0.915, 0.92, 0.932, 0.95, 0.954, 0.965,  0.966]
mistral_acc = [0.928, 0.94, 0.93, 0.937, 0.941, 0.95, 0.969,  0.97]

plot_pic_2(x_lst, mistral, qwen, llama, deepseek, mistral_acc,qwen_acc, llama_acc,deepseek_acc,'fig4.png')


def plot_pic_42(save_path):
    plt.figure(figsize=(3.6, 2.3))
    countries = ['Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']

    a = [0.8900, 0.9125, 0.9200]
    b = [0.8480, 0.8650, 0.8700]
    c = [0.8500, 0.8750, 0.8825]

    mean_values = [np.mean([a[i], b[i], c[i]]) for i in range(len(countries))]
    x = np.arange(len(countries))
    width = 0.2
    x_a = x
    x_b = x + width
    x_c = x + width * 2

    plt.bar(x_a, a, width=0.18, color="#9DC3E6", label='Jaccard', hatch='///', edgecolor='white')
    plt.bar(x_b, b, width=0.18, color="#2878B5", label='TF-IDF')  # hatch='...', edgecolor='white')
    plt.bar(x_c, c, width=0.18, color="#08509C", label='sentenceBERT')  # hatch='/', edgecolor='white')

    plt.plot(x + width, mean_values,
             color='black',
             marker='*',
             markersize=5,
             linewidth=1,
             linestyle='-',
             label='mean')

    plt.xticks(x + width, labels=countries, fontsize=5.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(ncol=4, fontsize=5.5, loc='upper right')  # fontsize=10,
    plt.ylim(0.6, 1)
    plt.yticks(fontsize=5.5)
    plt.ylabel('ASR', fontsize=5.5) # fontsize=12
    plt.savefig(save_path , dpi=300)

plot_pic_42('fig5.png')