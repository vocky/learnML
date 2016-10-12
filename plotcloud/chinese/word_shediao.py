#coding: utf-8

import os
from os import path
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import jieba

d = path.dirname(__file__)

def stopword(filename = ''):
    words = {}
    with open(filename, 'r') as f:
        line = f.readline().rstrip()
        while line:
            words.setdefault(line, 0)
            words[line.decode('utf-8')] = 1
            line = f.readline().rstrip()
    return words

stopwords = stopword(filename = './stopwords.txt')

def get_novel_names():
    name_path = './resources/names.txt'
    with open(name_path) as f:
        # 去掉结尾的换行符
        data = [line.strip().decode('gbk') for line in f.readlines()]
    novels = data[::2]
    names = data[1::2]
    novel_names = {k: v.split() for k, v in zip(novels, names)}
    return novel_names

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

novel_names = get_novel_names()
add2jieba(novel_names)

def get_book(filepath):
    with open(filepath) as f:
    	data = [line.decode('gbk', 'ignore').strip() 
		for line in f.readlines() 
		if line.decode('gbk', 'ignore').strip()]
        text = r' '.join(data)
        seg_generator = jieba.cut(text, cut_all=False)
        seg_list = [i for i in seg_generator if i not in stopwords]
        seg_list = [i for i in seg_list if i != u' ']
        seg_list = r' '.join(seg_list)
    return seg_list

sentences = get_book("./resources/books/射雕英雄传.txt")
print len(sentences)

china_mask = np.array(Image.open(path.join(d, "china_mask_1.png")))
wc = WordCloud(font_path='./simheittf/simhei.ttf', background_color="white", 
              margin=5, width=1800, height=800, mask=china_mask)
wc = wc.generate(sentences)
wc.to_file(path.join(d, "shediao.png"))

# show
plt.figure()
plt.imshow(wc)
plt.axis("off")
plt.show()
