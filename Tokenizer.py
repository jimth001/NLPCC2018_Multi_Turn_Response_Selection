# coding:utf-8
import thulac

class Tokenizer:
    def __init__(self):
        self.user_dict=None
        self.model_path=None#默认为model_path
        self.T2S=True#繁简体转换
        self.seg_only=True#只进行分词
        self.filt=True#去停用词
        self.tokenizer=thulac.thulac(user_dict=self.user_dict,model_path=self.model_path,T2S=self.T2S,seg_only=self.seg_only,filt=self.filt)

    def parser(self,text):
        return self.tokenizer.cut(text,text=True)#返回文本