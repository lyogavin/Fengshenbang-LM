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
        if item['title_type'] == 'step2_high_inter_1229':
            postfix_prompted_content = f"]\n小红书标题：[{item['title']}]"
            prompted_content = f"根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[{item['content']}"
        elif item['title_type'] == 'step3_template_1229':
            postfix_prompted_content = f"]\n最适合的标题类型：标题模版（根据句式[{item['title_template_name']}]改写标题）。\n小红书标题：[{item['title']}]"
            prompted_content = f"根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[{item['content']}"
        elif item['title_type'] == 'step4_suspences_1229':
            postfix_prompted_content = f"]\n最适合的标题类型：构造悬念（强调结果，保留部分信息，引发好奇）。\n小红书标题：[{item['title']}]"
            prompted_content = f"根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[{item['content']}"
        else:
            assert False

        postfix_input_ids = self.tokenizer.encode(postfix_prompted_content)

        prefix_input_ids = self.tokenizer.encode(prompted_content, truncation=True, max_length=self.max_seq_length - len(postfix_input_ids))


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
    datafile = '/home/ubuntu/cloudfs/ghost_data/combined_high_inter_template_suspences/combined_high_inter_template_suspences_val_sample_1229_1672366480.csv'
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


    for i in range(100, 24, -1):
        print(f"\nfor max len: {i}")
        testml.max_seq_length = i

        res = testml.encode({"title_type":"step2_high_inter_1229", 'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':'将数据转换成模型训练的输入'})

        print(f"encode len: {res['input_ids'].shape}")
        deres = testml.tokenizer.decode(res['input_ids'])
        print(f"decoded encode: {deres}")


    for i in range(100, 40, -1):
        print(f"\nfor max len: {i}")
        testml.max_seq_length = i

        res = testml.encode({"title_type":"step3_template_1229",
                             "title_template_name":"挑剔模版1",
                             'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':'将数据转换成模型训练的输入'})

        print(f"encode len: {res['input_ids'].shape}")
        deres = testml.tokenizer.decode(res['input_ids'])
        print(f"decoded encode: {deres}")


    for i in range(100, 40, -1):
        print(f"\nfor max len: {i}")
        testml.max_seq_length = i

        res = testml.encode({"title_type":"step4_suspences_1229", 'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':'将数据转换成模型训练的输入'})

        print(f"encode len: {res['input_ids'].shape}")
        deres = testml.tokenizer.decode(res['input_ids'])
        print(f"decoded encode: {deres}")
