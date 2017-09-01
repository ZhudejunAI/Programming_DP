# coding=utf-8

import json
import codecs

# fw = codecs.open('data.txt', 'w', encoding='utf-8')
fr = open('data.json', 'r')

for line in fr.readlines():
    json_dict = json.load(line)
    temp_msg = json_dict['msg']
    temp_msg.strip('\n')
    temp_msg.strip('\t')
    temp_msg.strip()
    # fw.write(temp_msg + '\n')
fr.close()
# fw.close()