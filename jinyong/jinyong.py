# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import jieba
import gensim
import scipy.cluster.hierarchy as sch


font_name = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
zhfont = mpl.font_manager.FontProperties(fname=font_name)

def show_chinese():
    x = range(10)
    plt.plot(x)
    plt.title(u'中文', fontproperties=zhfont)
    plt.show()

def get_novel_names():
    name_path = './resources/names.txt'
    with open(name_path) as f:
        # 去掉结尾的换行符
        data = [line.strip().decode('gbk') for line in f.readlines()]
    novels = data[::2]
    names = data[1::2]
    novel_names = {k: v.split() for k, v in zip(novels, names)}
    return novel_names

def plot_main_charecters(novel_names, novel, num=12):
    with open('./resources/books/{}.txt'.format(novel)) as f:
        data = f.read().decode('gbk', 'ignore')
    count = []
    for name in novel_names[novel]:
        count.append([name, data.count(name)])
    count.sort(key=lambda x: x[1])
    _, ax = plt.subplots()
    
    numbers = [x[1] for x in count[-num:]]
    names = [x[0] for x in count[-num:]]
    ax.barh(range(num), numbers, color='red', align='center')
    ax.set_title(novel, 
                 fontsize=14, 
                 fontproperties=zhfont)
    ax.set_yticks(range(num))
    ax.set_yticklabels(names, 
                       fontsize=14,
                       fontproperties=zhfont)
    plt.show()

def add2jieba(novel_names):
    for _, names in novel_names.iteritems():
        for name in names:
            jieba.add_word(name)
    with open("./resources/kungfu.txt") as f:
        kungfu_names = [line.decode('gbk').strip() 
                        for line in f.readlines()]
    with open("./resources/bangs.txt") as f:
        bang_names = [line.decode('gbk').strip() 
                      for line in f.readlines()]
    for name in kungfu_names:
        jieba.add_word(name)
    for name in bang_names:
        jieba.add_word(name)

def get_word2vec_mode(novel_names):
    # add all key words to jieba
    add2jieba(novel_names)
    # use jieba to cut words
    sentences = []
    for novel in novel_names.keys():
        print "处理：{}".format(novel)
        with open('./resources/books/{}.txt'.format(novel)) as f:
            data = [line.decode('gbk', 'ignore').strip() 
                    for line in f.readlines() 
                    if line.decode('gbk', 'ignore').strip()]
            for line in data:
                words = list(jieba.cut(line, cut_all=False))
                sentences.append(words)
    # build word2vec model
    model = gensim.models.Word2Vec(sentences, 
                                   size=100, 
                                   window=10, 
                                   min_count=20, 
                                   workers=4)
    return model

def print_similar(model, name_list):
    print 'similar to:', ','.join(name_list)
    for k, s in model.most_similar(positive=name_list, topn=5):
        print k, s

def plot_cluster(model, novel_names, novel):
    all_names = np.array([])
    word_vectors = None
    for name in novel_names[novel]:
        if name in model:
            all_names = np.append(all_names, name)
            if word_vectors is None:
                word_vectors = model[name]
            else:
                word_vectors = np.vstack((word_vectors, model[name]))
                all_names = np.array(all_names)
    Y = sch.linkage(word_vectors)
    _, ax = plt.subplots(figsize=(16, 12))
    Z = sch.dendrogram(Y, orientation='left')
    idx = Z['leaves']
    ax.set_xticks([])
    ax.set_yticklabels(all_names[idx], 
                      fontproperties=zhfont)
    ax.set_frame_on(False)
    plt.show()
    
if __name__ == '__main__':
    novel_names = get_novel_names()
    #plot_main_charecters(novel_names, '鹿鼎记')
    model = get_word2vec_mode(novel_names)
    print_similar(model, ["乔峰", "萧峰"])
    print_similar(model, ["双儿"])
    print_similar(model, ["张无忌", "赵敏"])
    print_similar(model, ["王语嫣"])
    print_similar(model, ["杨过"])
    print_similar(model, ["双儿", "方怡", "阿珂"])
    print_similar(model, ["丐帮"])
    print_similar(model, ["降龙十八掌"])
    plot_cluster(model, novel_names, "笑傲江湖")
