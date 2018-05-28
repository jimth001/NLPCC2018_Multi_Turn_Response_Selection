# coding:utf-8
import csv
import os
import codecs

class WordDictionary:
    def __init__(self):
        self.words = {}

    def add_word(self, word):
        if not word in self.words:
            self.words[word] = len(self.words)

    def get_index(self, word):
        if word in self.words:
            return self.words[word]
        else:
            return None

    def get_size(self):
        return len(self.words)

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        file = codecs.open(path + name, 'w+', encoding='utf-8')
        wr = csv.writer(file)
        for key in self.words.keys():
            wr.writerow([key, self.words[key]])
        file.close()

    def load(self, path, name):
        self.words.clear()
        file = codecs.open(path + name, 'r', encoding='utf-8')
        rd = csv.reader(file)
        for strs in rd:
            self.words[strs[0]] = int(strs[1])
        file.close()
