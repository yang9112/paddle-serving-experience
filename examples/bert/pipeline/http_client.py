# -*- coding: utf-8 -*-

import json
import requests

url = "http://127.0.0.1:18082/code-review/prediction"

data = {"key": ["0"],
        "value": ["国有企业引入非国有资本对创新绩效的影响——基于制造业国有上市公司的经验证据"]}

r = requests.post(url=url, data=json.dumps(data))
print(r.json())
