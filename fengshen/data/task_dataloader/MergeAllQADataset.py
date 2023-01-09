# coding=utf8
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
import torch

class GPT2QADataset(Dataset):
    '''
    Dataset Used for yuyuan medical qa task.
    Just surpport small datasets, when deal with large datasets it may be slowly.
    for large datasets please use mmapdatasets(doing)
    '''

    def __init__(self, data_path, name, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
        self.data_size = os.path.getsize(data_path)/1024/1024/1024
        self.data_type_name = name
        self.data = self.load_data(data_path)
        self.max_seq_length = args.max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.encode(self.data.iloc[index])

    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        return df



    def load_data_file(self, data_path):
        # 有进度条展示
        if self.data_size <= 5:
            with open(data_path, "rt", encoding='utf8') as f:
                lines = f.readlines()
            total_num = len(lines)
            data_gen = lines
        else:
            data_gen = open(data_path, "rt", encoding='utf8')
            total_num = None

        data = []
        with tqdm(total=total_num, desc=f'{self.data_type_name}处理进度', mininterval=0.3) as bar:
            for idx, line in enumerate(data_gen):
                data.append(self.data_parse(line))
                bar.update()

        if self.data_size > 5:
            data_gen.close()
        return data

    def data_parse(self, line):
        """
        解析不同格式的数据
        """
        dic = eval(line.strip())
        return dic

    def encode(self, item):
        """
        将数据转换成模型训练的输入
        """

        def get_promote_title_type(row):
            if row['data_type'] in ['redbook_tags_title', 'redbook_tags_title_top2_sim']:
                return f"小红书"
            elif row['source_category'] in ['newrank_healthcare', 'newred_0905','redbook_0707', 'redbook_1228']:
                return f"小红书"
            elif row['source_category'] in ['newred_0905_ocr']:
                return f"封面图片"
            elif row['source_category'] in ['baidubaijia', 'newrank_hot_wx_account_0929', 'sougou_wx', 'sougou_wx_search_expansion','zhihu_search_exp_article']:
                return f"公众号"
            elif row['source_category'] in ['xunyiwenyao','snowball']:
                return f""
            elif row['source_category'] in ['zhihu_search', 'zhihu_fin_topic_download','zhihu_search_exp_question']:
                return f"知乎"
            elif row['source_category'] in ['taobao_jd']:
                return f"电商"
            else:
                assert False, f"{row['data_type']}, {row['source_category']}"

        truncated_title = item['title']
        truncated_title = truncated_title[:120] if isinstance(truncated_title,str) else ' '
        postfix_prompted_content = f"]\n符合以上内容的{get_promote_title_type(item)}标题：[{truncated_title}]\n"

        prompted_content = f"你是自媒体创作者，需要根据指定内容，撰写适合指定平台的爆款标题。\n自媒体创作者：\n需要起标题的内容：[{item['content']}"



        postfix_input_ids = self.tokenizer.encode(postfix_prompted_content)

        max_length_left = self.max_seq_length - len(postfix_input_ids)
        max_length_left = max(3,max_length_left)
        #print(f"max_length_left:{max_length_left}")
        prefix_input_ids = self.tokenizer.encode(prompted_content, truncation=True, max_length=max_length_left)


        input_ids = torch.tensor([prefix_input_ids + postfix_input_ids])
        attention_mask = torch.tensor([[1] * (len(postfix_input_ids) + len(prefix_input_ids))])

        #inputs_dict = self.tokenizer.encode_plus(item['prompted_content'],
        #                                         max_length=self.max_seq_length, padding='max_length',
        #                                         truncation=True, return_tensors='pt')
        #prompt_inputs_dict = self.tokenizer.encode_plus(item['prompt'],
        #                                         max_length=self.max_seq_length, padding=False,
        #                                         truncation=True, return_tensors='pt')
        target = input_ids
        labels = target.clone().detach()
        labels[target == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "labels": labels.squeeze(),
            "labels_mask": torch.tensor([[0] * len(prefix_input_ids) + [1] * len(postfix_input_ids)])
            #"answer_end_pos":len(answer_input_ids) + len(postfix_input_ids) + len(prefix_input_ids) - 1
        }


class GPT2QADataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('GPT2QADataModel')
        parser.add_argument('--data_dir', type=str, required=True)
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--train_data', default='train.txt', type=str)
        parser.add_argument('--valid_data', default='valid.txt', type=str)
        parser.add_argument('--test_data', default='test.txt', type=str)
        parser.add_argument('--train_batchsize', type=int, required=True)
        parser.add_argument('--valid_batchsize', type=int, required=True)
        parser.add_argument('--max_seq_length', default=1024, type=int)
        return parent_args

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_batchsize = args.train_batchsize
        self.valid_batchsize = args.valid_batchsize
        if not args.do_eval_only:
            self.train_data = GPT2QADataset(os.path.join(
                args.data_dir, args.train_data), '训练集', args)
            self.valid_data = GPT2QADataset(os.path.join(
                args.data_dir, args.valid_data), '验证集', args)
        self.test_data = GPT2QADataset(os.path.join(
            args.data_dir, args.test_data), '测试集', args)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, shuffle=True,
            batch_size=self.train_batchsize,
            pin_memory=False, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False,
                          batch_size=self.valid_batchsize,
                          pin_memory=False, num_workers=self.args.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, shuffle=False,
                          batch_size=self.valid_batchsize, pin_memory=False,
                          num_workers=self.args.num_workers)


if __name__ == '__main__':
    import argparse
    #modelfile = '/cognitive_comp/wuziwei/pretrained_model_hf/medical_v2'
    datafile = '/home/ubuntu/cloudfs/ghost_data/merge_all_add_1208_1228//merge_all_0108_val_sample_1673194850.csv'
    parser = argparse.ArgumentParser(description='hf test', allow_abbrev=False)
    group = parser.add_argument_group(title='test args')
    group.add_argument('--pretrained-model-path', type=str, default="/home/ubuntu/cloudfs/saved_models/bigscience/bloomz-3b",
                       help='Number of transformer layers.')
    group.add_argument('--max-seq-length', type=int, default=1024)
    args = parser.parse_args()

    testml = GPT2QADataset(datafile, 'medical_qa', args=args)

    print(testml[10])
    print(len(testml))
    print(f"max len:{testml.max_seq_length}")

    for testing_type in ["newrank_healthcare", "zhihu_search", "baidubaijia"]:



        for i in range(100, 40, -1):
            print(f"\nfor max len: {i}")
            testml.max_seq_length = i

            res = testml.encode({"data_type":"redbook_content_title",
                                 "source_category":testing_type, 'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':'将数据转换成模型训练的输入'})

            print(f"encode len: {res['input_ids'].shape}")
            deres = testml.tokenizer.decode(res['input_ids'])
            print(f"decoded encode: {deres}")


    for title_len in range(80, 150):
        title = ['数'] * title_len
        title = ''.join(title)
        testml.max_seq_length = 200
        print(f"title len:{title_len}")

        res = testml.encode({"data_type":"redbook_content_title",
                             "source_category":"newrank_healthcare",
                             'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':title})

        print(f"encode len: {res['input_ids'].shape}")
        deres = testml.tokenizer.decode(res['input_ids'])
        print(f"decoded encode: {deres}")