# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import jieba
import gensim

font_name = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
zhfont = mpl.font_manager.FontProperties(fname=font_name)

def show_chinese():
    x = range(10)
    plt.plot(x)
    plt.title(u'中文', fontproperties=zhfont)
    plt.show()

name_path = './resources/names.txt'
with open(name_path) as f:
    # 去掉结尾的换行符
    data = [line.strip().decode('gbk') for line in f.readlines()]

novels = data[::2]
names = data[1::2]

novel_names = {k: v.split() for k, v in zip(novels, names)}

#for name in novel_names['天龙八部'][:20]:
#    print name

def find_main_charecters(novel, num=12):
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
    
sentences = []
for novel in novels:
    print "处理：{}".format(novel)
    with open('./resources/books/{}.txt'.format(novel)) as f:
        data = [line.decode('gbk', 'ignore').strip() 
                for line in f.readlines() 
                if line.decode('gbk', 'ignore').strip()]
        for line in data:
            words = list(jieba.cut(line, cut_all=False))
            sentences.append(words)

model = gensim.models.Word2Vec(sentences, 
                               size=100, 
                               window=6, 
                               min_count=10, 
                               workers=4)

for k, s in model.most_similar(positive=["乔峰", "萧峰"]):
    print k, s

for k, s in model.most_similar(positive=["阿朱"]):
    print k, s

#if __name__ == '__main__':
#    find_main_charecters('鹿鼎记')
 