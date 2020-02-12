# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:20

from keras_bert import extract_embeddings

model_path = 'chinese_L-12_H-768_A-12'
texts = ['今晚（2月11日），钟南山院士接受总台央视记者独家专访，通过央视回应了近日媒体报道“钟南山的最新论文发现新冠肺炎潜伏期最长可达24天”的问题。',
         '英国苏格兰政府首席大臣、苏格兰民族党党魁妮古拉·斯特金11日在伦敦说，苏格兰人应有权重新选择是否独立。',
         '教育部门明确提出“延期开学”是假期的延续，各校均不得以任何形式集体组织上新课，也不得举行任何形式的线下教学活动和集体活动。'
         ]

embeddings = extract_embeddings(model_path, texts)

print(type(embeddings))
print(embeddings)

for _ in embeddings:
    print(_[0].shape)