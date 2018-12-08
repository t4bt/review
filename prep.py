import mysql.connector
import MeCab
import numpy as np
import gensim
import csv
import pickle
from Ocab import Ocab, Regexp

class load:

    @staticmethod    #mysqlデータベースからのselect
    def database(db_settings, sql):
        con = mysql.connector.connect(**db_settings)
        cur = con.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        con.close()
        return result

    @staticmethod
    def sentences(fname='corpus.csv', padding=False):
        # ファイルを読み込む
        f = open(fname, 'r')

        # 全ての文
        sentence = []

        # 1行ずつ処理する
        line = f.readline()
        while line:
            one = list(map(int,line.split(',')))
            one.append(-2)    #穴埋めの-2を追加
            sentence.append(one)    #新しい文を追加
            line = f.readline()
        f.close()

        # 単語の種類
        # no_belowとno_aboveで省いた文字(-1)と穴埋め文字(-2)の2つを追加
        n_words = max([max(l) for l in sentence]) + 3
        print("n_words:{}".format(n_words))
        # 最長の文の長さ
        l_max = max([len(l) for l in sentence])
        print("length:{}".format(l_max))

        UNK = n_words - 2
        EOS = n_words - 1
        for i, s in enumerate(sentence):
            sentence[i] = [EOS if j==-2 else UNK if j==-1 else j for j in s]

        if padding:
            # 穴埋めする
            for i in range(len(sentence)):
                sentence[i].extend([EOS]*(l_max-len(sentence[i])))
            return np.array(sentence), n_words
        else:
            for i in range(len(sentence)):
                sentence[i] = np.array(sentence[i][:-1], np.int32)
            return np.array(sentence), n_words-1

    @staticmethod
    def wakati2vector(wakati_file, vector_file, dictionary="fasttext_dict"):
        model = gensim.models.KeyedVectors.load_word2vec_format(vector_file)
        i2w = model.wv.index2word
        w2i = {w: i for i, w in enumerate(i2w)}
        with open(dictionary, "wb") as f:
            pickle.dump((i2w, w2i), f, protocol=4)
        n_words = len(i2w)
        v_size = model.vector_size
        initialW = model.wv.vectors
        f = open(wakati_file, "r")
        sentence = []
        line = f.readline()
        while line:
            one = line.split(" ")[:-1]
            one = np.array([w2i[w] for w in one], np.int32)
            sentence.append(one)
            line = f.readline()
        f.close()

        l_max = max([len(l) for l in sentence])
        print("n_words:{}".format(n_words))
        print("length:{}".format(l_max))

        return np.array(sentence), n_words, v_size, initialW


class wakati:

    @staticmethod    #文章のリストを分かち書きする
    def wakati(text_list, fname='wakati.txt', get_hinsi=[], save=True):
        # 分かち書き
        if len(get_hinsi) == 0:
            c = Regexp()
            m = MeCab.Tagger("-Owakati")
            wakati_text = [m.parse(c.normalize(text).lower()) for text in text_list]
        else:
            c = Regexp()
            m = MeCab.Tagger()
            def _parse(text):
                output = ""
                for row in m.parse(c.normalize(text).lower()).split('\n'):
                    if row == 'EOS':
                        break
                    word, *hinsi = row.split('\t')
                    hinsi = hinsi[0].split(',')[0]
                    if hinsi in get_hinsi:
                        output += word + ' '
                if not output == "":
                    output += '\n'
                return output
            wakati_text = [_parse(text) for text in text_list]

        # 出力
        if not save:    #ファイル保存しなければ分かち書きしたリストを返す
            return wakati_text
        else:            #デフォルトでファイルを出力
            with open(fname, "w") as f:
                for text in wakati_text:
                    f.write(text)

    @staticmethod
    def wakati_char(text_list, fname='wakati_char.txt', save=True):
        wakati_text = []
        for text in text_list:
            tmp = text.split(' ')
            wakati_char = ''
            for sentence in tmp:
                for c in sentence:
                    wakati_char += c+' '
            wakati_text.append(wakati_char+'\n')
        if not save:
            return wakati_text
        else:
            with open(fname, "w") as f:
                for text in wakati_text:
                    f.write(text)

    @staticmethod
    def wakati_normalize(text_list, fname="wakati.txt", save=True):
        c = Regexp()
        m = Ocab(target=["名詞","動詞","形容詞","副詞"])
        def normalize(text):
            text1 = c.normalize(text)
            text2 = m.wakati(text1)
            text3 = m.removeStoplist(text2, [])
            return text3
        wakati_text = [normalize(text) for text in text_list]
        # 出力
        if not save:    #ファイル保存しなければ分かち書きしたリストを返す
            idx = []
            text_list_ = []
            for i, text in enumerate(wakati_text):
                if not text == '':
                    idx.append(i)
                    text_list_.append(str(text)+" \n")
            return text_list_, idx
        else:            #デフォルトでファイルを出力
            idx = []
            text_list_ = []
            with open(fname, "w") as f:
                for i, text in enumerate(wakati_text):
                    if not text == '':
                        idx.append(i)
                        text_list_.append(text)
                        f.write(str(text)+' \n')
            return text_list_, idx


class Wakati2Corpus:

    def __init__(self, no_below=20, no_above=1):
        self.no_below = no_below
        self.no_above = no_above

    #分かち書きのファイルの各単語を単語インデックスへ変換
    def wakati2index(self, fname):
        corpus, dictionary = self.create_corpus(fname, mode='idx')
        dictionary.save_as_text("dictionary.txt")
        with open("corpus.csv","w") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(corpus)

    def wakati2dict(self, fname="wakati.txt"):
        c = 0
        texts_words = {}
        with open(fname, 'r') as f:
             line = f.readline()
             while line:
                 texts_words[c] = line.split()
                 c += 1
                 line = f.readline()
        return texts_words

    def create_dict(self, texts_words):
        dictionary = gensim.corpora.Dictionary(texts_words.values())
        dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        return dictionary

    def create_corpus(self, fname="wakati.txt", mode='bow'):
        texts_words = self.wakati2dict(fname)
        dictionary = self.create_dict(texts_words)
        if mode == 'bow':
            corpus = [dictionary.doc2bow(words) for words in texts_words.values()]
        elif mode == 'idx':
            corpus = [dictionary.doc2idx(words) for words in texts_words.values()]
        return corpus, dictionary
