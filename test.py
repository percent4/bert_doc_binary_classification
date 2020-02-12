# -*- coding: utf-8 -*-
# author: Jclian91
# place: Pudong Shanghai
# time: 2020-02-12 12:45
from bert.extract_feature import BertVector

bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=100)

import time
t1 = time.time()
for _ in range(100):
    print(_)
    text = ['英国苏格兰政府首席大臣、苏格兰民族党党魁妮古拉·斯特金11日在伦敦说，苏格兰人应有权重新选择是否独立。']*1000
    vec = bert_model.encode(text)["encodes"][0]

t2 = time.time()
print(t2 - t1)