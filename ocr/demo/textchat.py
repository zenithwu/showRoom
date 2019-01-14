import optparse
import time
#这里我上端代码独立生成一个文件“apiutil.py"，所以要导入一下
import json

from ocr.demo import apiutil

app_key = 'SBmTSxUWAJZaOauE'
app_id = '2110757327'
questionS= '你好吗？'
def anso(questionS):
    str_question = questionS
    session = 10000
    ai_obj = apiutil.AiPlat(app_id, app_key)

    rsp = ai_obj.getNlpTextChat(session,str_question)
    if rsp['ret'] == 0:
        print('............................................................')
        ask = (rsp['data'])['answer']
        print(ask)
    else:
        print(json.dumps(rsp, ensure_ascii=False, sort_keys=False, indent=4))
if __name__ == '__main__':
    anso(questionS)
