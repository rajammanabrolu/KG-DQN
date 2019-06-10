import re
from nltk.tokenize import sent_tokenize
from fuzzywuzzy import fuzz
import json
import random

data_list = []

with open('./oracle.txt', 'r') as f:
    cur = []
    cur_admissible_actions = "N, S, E, W, look, examine"
    cur_taken_action = ""
    id = 0

    for line in f:
        line = line.replace('\n', '')
        if line != '---------' and "Actions:" not in str(line) and "Taken action:" not in str(line):
            cur.append(line)
        elif "Actions:" in str(line):
            cur_admissible_actions = str(line).split(':')[1].replace("'", '').replace("[", '').replace("]", '')
            #print(cur_admissible_actions)
        elif "Taken action:" in str(line):
            cur_taken_action = str(line).split(':')[1]
            #print(cur_taken_action)
        elif line == '---------':
            cur = [a.strip() for a in cur]
            cur = ' '.join(cur).strip().replace('\n', '').replace('---------', '')
            try:
                title = re.findall("(?<=-\=).*?(?=\=-)", cur)[0].strip()
            except IndexError:
                title = "UNK"
            cur = re.sub("(?<=-\=).*?(?=\=-)", '', cur)
            cur = cur.replace("-==-", '').strip()
            cur = '. '.join([a.strip() for a in cur.split('.')])
            #print(cur)

            answer_list = []

            for sent in sent_tokenize(cur):
                if fuzz.token_set_ratio(cur_taken_action, sent) > 60:
                    #print(fuzz.token_set_ratio(cur_taken_action, sent), fuzz.partial_ratio(cur_taken_action, sent), cur_taken_action, sent)
                    answer_start = cur.find(sent)
                    answer_item = {"answer_start": answer_start, "text": sent}
                    answer_list.append(answer_item)

            cur = cur + "The actions are: " + str(cur_admissible_actions) + "."
            answer_list.append({"answer_start": cur.find(cur_taken_action), "text": cur_taken_action})

            qa_item = {"answers": answer_list, "question": "What action should I take?", "id": str(id)}
            id += 1
            qa_list = [qa_item]

            paragraph_item = {"context": cur + "The actions are: " + str(cur_admissible_actions) + ".", "qas": qa_list}
            paragraph_list = [paragraph_item]
            data_item = {"title": title, "paragraphs": paragraph_list}
            data_list.append(data_item)
            #print(data_item)


            cur = []
            cur_admissible_actions = ""
            cur_taken_action = ""

out_train = open('./cleaned_qa_train.json', 'w')
out_dev = open('./cleaned_qa_dev.json', 'w')

llist = len(data_list)
random.shuffle(data_list)

data_train = {"data": data_list[:int(llist * 0.9)]}
data_dev = {"data": data_list[int(llist * 0.9):]}

json.dump(data_train, out_train)
json.dump(data_dev, out_dev)
out_train.close()
out_dev.close()
