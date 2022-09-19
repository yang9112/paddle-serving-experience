# -*- coding: utf-8 -*-

import time
import numpy as np

from paddle_serving_server.pipeline import PipelineClient

client = PipelineClient()
client.connect(['127.0.0.1:8088'])

list_data = [
    "国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据",
    "试论翻译过程中的文化差异与语言空缺翻译过程,文化差异,语言空缺,文化对比"
]
feed = {}
for i, item in enumerate(list_data):
    feed[str(i)] = item

print(feed)
start_time = time.time()
ret = client.predict(feed_dict=feed)
end_time = time.time()
print("time to cost :{} seconds".format(end_time - start_time))

result = np.array(eval(ret.value[0]))
print(result)