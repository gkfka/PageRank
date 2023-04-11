import json
import os
from textrank import KeywordSummarizer, KeysentenceSummarizer


def is_check(sent1, sent2):
    count = 0
    for i in sent1:

        if i in sent2:
            count += 1
    return count

def mrr_func(pred, true):
    for j in range(3):
        for i in range(3):
            if pred[j] == true[i]:
                if i == 0:  return 1
                elif i == 1: return 1/2
                elif i == 2: return 1/3
    return 0

def eval(pred_file, true_file):
    pred_sents = open(pred_file,'r',encoding='utf8').readlines()
    true_sents = open(true_file, 'r', encoding='utf8').readlines()
    correct = 0
    mrr_value = 0
    for s_idx, sents in enumerate(pred_sents):

        pred = [sent for  sent in sents.strip('\n').split('##')]
        true = [sent for sent in true_sents[s_idx].strip('\n').split('##')]

        correct += is_check(pred, true)
        mrr_value += mrr_func(pred, true)
    acc = correct /(len(pred_sents)*3)
    mrr = mrr_value / len(pred_sents)
    print("pre, recall  : ", acc, "mrr : ", mrr )



def read_data(fname):
    # json파일 읽음
    with open(fname, 'rt', encoding="UTF8") as f:
        json_data = json.load(f)
    document = []
    extractive = []
    for i in range(len(json_data['documents'])):
        sentence = []

        for e in json_data['documents'][i]['text']:
            for s in e:
                sentence.append(s['sentence'])

        extractive.append(json_data['documents'][i]['extractive'])

        document.append((sentence))
    return document, extractive

from konlpy.tag import Komoran

komoran = Komoran()

def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

def komoran_tokenize(sent):
    words = sent.split()
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words
def write_file(fname, sents):
    f = open(fname, 'w', encoding='utf8')
    for i in sents:
        f.write('##'.join(i))
        f.write('\n\n')
def write_rank_file(fname, sents):
    f = open(fname, 'w', encoding='utf8')
    for i in sents:
        f.write(' '.join(i))
        f.write('\n\n')

def experiment(out_file, true_file):

    bbb = 'D:/2022/그래프/과제1_문서요약/file/'
    aaa = os.listdir('D:/2022/그래프/과제1_문서요약/file')
    extract_file = open('extract.txt', 'w', encoding='utf8')
    # out_file = open('result.txt','w',encoding='utf8')

    doucement, extract = [], []
    summarizer = KeysentenceSummarizer(
        tokenize=komoran_tokenizer,
        min_sim=0.3,
        verbose=False
    )

    keyword_extractor = KeywordSummarizer(
        tokenize=komoran_tokenizer,
        window=-1,
        verbose=False
    )
    doc_ex = []


    for ia, dir in enumerate(aaa):

        # json_file = dir
        print(bbb+dir + "하는중~~\n")
        doc, ext = read_data(bbb+dir)

        if not len(doc_ex) > 0:
            doc_ex = doc[:10]

        for i, sents in enumerate(doc):
            try:
                keysents = summarizer.summarize(sents, topk=3)

                out = [sent for _, _, sent in keysents]
                cor = [doc[i][idx] for idx in ext[i]]
                doucement.append(out)
                extract.append(cor)
            except:
                continue


    for sent in doc_ex:
        keywords = keyword_extractor.summarize(sent, topk=30)
        keys = [word for word, _ in keywords]
        extract_file.write(' '.join(keys))
        extract_file.write('\n')

    write_file(out_file, doucement)
    write_file(true_file, extract)

if __name__ == '__main__':
    out_file = 'result.txt'
    true_file = 'correct.txt'
    experiment(out_file, true_file)
    eval(out_file, true_file)

