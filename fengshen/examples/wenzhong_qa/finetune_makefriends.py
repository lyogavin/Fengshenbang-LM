import sys

sys.path.append('../..')
print(sys.path)

from transformers import GPT2LMHeadModel
from data.task_dataloader.MakeFriendsQADataset import GPT2QADataModel
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
import torch
import os
import random
import sys
sys.path.insert(0, '/cognitive_comp/wuziwei/codes/fengshen/fengshen')
# sys.path.append('../../')
# sys.path.append('../')
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

# ad hoc

def generate_agent_paraphrase(model, tokenizer, encoded_input_ids, return_count=5, verbose=False,
                              diversity_enhanced=False, diversity_penalty=0.1,
                              bad_words_ids=None,
                              typical_p=0.95):


    model_generate_params = {
        'max_length': 800,
        # 'min_length':int(MAX_TARGET_LEN/3*2),
        'num_beams': 4,
        'num_return_sequences': 2,
        'temperature': 1.1,
        'output_scores': True,
        'repetition_penalty': 1.,  # https://arxiv.org/pdf/1909.05858.pdf,
        'return_dict_in_generate': True,
        'no_repeat_ngram_size': 4,
        'top_p': 0.85,
        # 'top_k':5,
        'do_sample': True,
        'early_stopping': True
    }

    if diversity_enhanced:
        model_generate_params = {**model_generate_params,
                                 **{
                                     'num_beam_groups': return_count,
                                     'diversity_penalty': diversity_penalty,
                                     'do_sample': False
                                 }}
    elif typical_p is not None:
        model_generate_params['typical_p'] = typical_p

    if bad_words_ids is not None:
        model_generate_params['bad_words_ids'] = bad_words_ids

    # get prompt score:
    outputs = model(input_ids=encoded_input_ids, labels=encoded_input_ids)

    prompt_score = outputs.loss.cpu().detach()
    torch.cuda.empty_cache()

    text_outputs = model.generate(input_ids=encoded_input_ids,
                                  **model_generate_params
                                  )  # ,attention_mask=attention_mask)
    decoded_text_outputs = tokenizer.batch_decode(text_outputs.sequences, skip_special_tokens=True)

    list_with_scores = list(zip([s.item() for s in text_outputs.sequences_scores],
                                [x.replace(" ", " ") for x in decoded_text_outputs]))
    return list_with_scores



class GPT2FinetuneMedicalQAModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./ckpt/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)
        parser.add_argument('--save_last', action='store_true', default=True)
        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_train_steps', default=1000, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)

        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         #  every_n_train_steps=args.every_n_train_steps,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename,
                                         save_last=args.save_last)

    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(*args, **kwargs)

class GPT2FinetuneMedicalQA(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--learning_rate', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        return parent_args

    def __init__(self, args, num_data, tokenizer=None):
        super().__init__()
        self.args = args
        self.num_data = num_data
        self.tokenizer = tokenizer
        print('num_data:', num_data)
        self.model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss)
        # self.log('val_acc', acc)
        #print(f"input ids : {batch['input_ids']}")

        return batch

    def validation_epoch_end(self, training_step_outputs):
        for bi, batch in enumerate(random.shuffle(training_step_outputs)[:10]):
            print(f"--{bi}/{10}--- input text: {batch['prompt']}")
            prediction = generate_agent_paraphrase(self.model, self.tokenizer, batch['prompt_input_ids'])
            print(f"validation_samples:\nlabels: {batch['prompted_content']}\npredictions: {prediction}")

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]


def main():
    total_parser = argparse.ArgumentParser("Summary Task")
    total_parser.add_argument(
        '--run_ts', default=None, type=str)
    total_parser.add_argument(
        '--do_eval_only', action='store_true', default=False)
    total_parser.add_argument(
        '--pretrained_model_path', default=None, type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json', type=str)
    # * Args for data preprocessing
    total_parser = GPT2QADataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = GPT2FinetuneMedicalQAModelCheckpoint.add_argparse_args(
        total_parser)
    total_parser = GPT2FinetuneMedicalQA.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()

    print(f"args:{args}")

    data_model = GPT2QADataModel(args)
    if not args.do_eval_only:
        model = GPT2FinetuneMedicalQA(args,
                                      len(data_model.train_dataloader()),
                                      tokenizer=data_model.valid_data.tokenizer
                                      )
        checkpoint_callback = GPT2FinetuneMedicalQAModelCheckpoint(
            args).callbacks
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'), name='MedicalQA-GPT2')
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             #evaluation_strategy='epoch',
                                             callbacks=[checkpoint_callback]
                                             )
        trainer.fit(model, data_model)

        # result = trainer.predict(model, data_model)
        # with open('test_results.txt', 'wt', encoding='utf-8') as w:
        #     for line in result:
        #         w.writelines(line)

        model.model.save_pretrained(os.path.join(args.default_root_dir, f"gpt2_all_merged_gen_content_finetune_{args.run_ts}"))
           # '/cognitive_comp/wuziwei/pretrained_model_hf')
        print(f'model saved: {os.path.join(args.default_root_dir, f"gpt2_all_merged_gen_content_finetune_{args.run_ts}")}')
    else:
        print('save to hf.....')
        trainer = Trainer.from_argparse_args(args)
        model = GPT2FinetuneMedicalQA(
            args, len(data_model.predict_dataloader()))

        result = trainer.predict(
            model, data_model, ckpt_path='/cognitive_comp/wuziwei/task/fs_medical_qa_finetune/ckpt/last.ckpt')
        # with open('test_results.txt','wt',encoding='utf-8') as w:
        #     for line in result:
        #         w.writelines(line)

        model.model.save_pretrained(
            '/cognitive_comp/wuziwei/pretrained_model_hf')


if __name__ == '__main__':
    main()
