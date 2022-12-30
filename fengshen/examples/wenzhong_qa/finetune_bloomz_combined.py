import sys

sys.path.append('../..')
print(sys.path)

from transformers import BloomForCausalLM
from data.task_dataloader.CombinedQADataset import GPT2QADataModel
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, loggers
from torch.nn import CrossEntropyLoss
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse
import torch
import os
import random
import sys
import logging
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
        'max_new_tokens': 30,
        # 'min_length':int(MAX_TARGET_LEN/3*2),
        'num_beams': 4,
        'num_return_sequences': 1,
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

def logits_labels_mask_to_loss(logits, labels, mask, verbose=False):
    # logits output:
    lm_logits = logits
    if verbose:
        print(f"logits shape: {logits}")

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_labels_mask = mask[..., 1:].contiguous()
    batch_size, seq_length, vocab_size = shift_logits.shape
    if verbose:
        print(f"shift_logits shape: {shift_logits.shape}")

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
    ).view(batch_size, seq_length)
    if verbose:
        print(f"loss shape: {loss.shape}")

    loss_masked = loss * shift_labels_mask
    loss = torch.mean(loss_masked)
    if verbose:
        print(f"mask: {mask}")
    if verbose:
        print(f"loss: {loss}")

    return loss

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
        self.model = BloomForCausalLM.from_pretrained(
            args.pretrained_model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)


    # !!!! following: https://github.com/MehwishFatimah/GPT2_Summarization/blob/7bd72af159d6859e12a7af646ec5435020de631a/modules/training.py#L88
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bloom/modeling_bloom.py#L864
    # only consider loss for the answering part!!!!
    def training_step(self, batch, batch_idx):
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'], labels=batch['labels'])


        loss = logits_labels_mask_to_loss(output.logits, batch['labels'], batch['labels_mask'])


        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', loss, on_epoch=True)
        return loss

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


        loss = logits_labels_mask_to_loss(output.logits, batch['labels'], batch['labels_mask'])

        self.log('val_loss', loss, sync_dist=True, on_epoch=True)
        # self.log('val_acc', acc)
        #print(f"input ids : {batch['input_ids']}")

        return batch

    def training_epoch_end(self, training_step_outputs):
        print(f"\ntraining_epoch_end...")
        gathered = self.all_gather(training_step_outputs)
        if self.global_rank == 0:
            # print(gathered)
            loss = sum(output['loss'].mean() for output in gathered) / len(training_step_outputs)
            print(f"train loss:{loss.item()}")

    def validation_epoch_end(self, training_step_outputs):
        print(f"\nvalidation_epoch_end...")
        gathered = self.all_gather(training_step_outputs)
        if self.global_rank == 0:
            # print(gathered)
            loss = sum(output['loss'].mean() for output in gathered) / len(training_step_outputs)
            print(f"val loss:{loss.item()}")


        #random.shuffle(training_step_outputs)
        #for bi, batch in enumerate(training_step_outputs):
        #    print(f"--{bi}/{len(training_step_outputs)}--- input text: {batch['prompt']}")
        #    prediction = generate_agent_paraphrase(self.model, self.tokenizer, batch['prompt_input_ids'])
        #    print(f"validation_samples:\nlabels: {batch['prompted_content']}\npredictions: {prediction}")

        torch.cuda.empty_cache()
        additional_cases = ["根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[暗黑系美甲]\n最适合的标题类型：构造悬念。\n小红书标题：[",
                            "根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[扫地机器人]\n最适合的标题类型：构造悬念。\n小红书标题：[",
                            "根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[草原羊肉]\n最适合的标题类型：构造悬念。\n小红书标题：[",
                            "根据指定内容，撰写爆款小红书笔记标题。\n需要起标题的内容：[真无线蓝牙耳机]\n最适合的标题类型：构造悬念。\n小红书标题：["]
        for bi, item in enumerate(additional_cases):
            print(f"--{bi}/{len(additional_cases)}--- input text: {item}")

            prompt_inputs_dict = self.tokenizer.batch_encode_plus([item],
                                                            max_length=100, padding=False,
                                                            truncation=True, return_tensors='pt')

            prompt_inputs_ids = prompt_inputs_dict['input_ids'].to("cuda")
            #print(f"shape prompt_inputs_ids: {prompt_inputs_ids.shape}")
            prediction = generate_agent_paraphrase(self.model, self.tokenizer, prompt_inputs_ids)
            print(f"validation_samples:\npredictions: {prediction}")

            torch.cuda.empty_cache()

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


def main(args):
    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")

    logger.addHandler(logging.StreamHandler(sys.stdout))


    data_model = GPT2QADataModel(args)
    if not args.do_eval_only:
        model = GPT2FinetuneMedicalQA(args,
                                      len(data_model.train_dataloader()),
                                      tokenizer=data_model.valid_data.tokenizer
                                      )
        checkpoint_callback = GPT2FinetuneMedicalQAModelCheckpoint(
            args).callbacks
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'), name='bloomz_combined')

        #wanddb_logger = loggers.WandbLogger(save_dir=os.path.join(
        #    args.default_root_dir, 'log/'), name='bloomz_combined')
        csv_logger = loggers.CSVLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'), name='bloomz_combined')
        trainer = Trainer.from_argparse_args(args,
                                             logger=[logger,csv_logger],
                                             #evaluation_strategy='epoch',
                                             callbacks=[checkpoint_callback]
                                             )
        trainer.fit(model, data_model)

        # result = trainer.predict(model, data_model)
        # with open('test_results.txt', 'wt', encoding='utf-8') as w:
        #     for line in result:
        #         w.writelines(line)

        model.model.save_pretrained(os.path.join(args.default_root_dir, f"bloomz_combined_finetune_{args.run_ts}"))
           # '/cognitive_comp/wuziwei/pretrained_model_hf')
        print(f'model saved: {os.path.join(args.default_root_dir, f"bloomz_combined_finetune_{args.run_ts}")}')
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



def test():
    from transformers import AutoTokenizer
    # from fengshen.examples.pegasus.tokenizers_pegasus import PegasusTokenizer
    # from transformers import PegasusForConditionalGeneration
    from transformers import BloomForCausalLM


    TOKENIER_MODEL_NAME = '/home/ubuntu/cloudfs/saved_models/bigscience/bloomz-3b'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIER_MODEL_NAME)

    USE_PROXY = False
    from transformers import BloomForCausalLM

    # model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    # model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    if USE_PROXY:
        model = BloomForCausalLM.from_pretrained(TOKENIER_MODEL_NAME,
                                                 proxies={'http': '43.159.130.13:8123'})
    else:
        model = BloomForCausalLM.from_pretrained(TOKENIER_MODEL_NAME)

    MODEL_NAME = '/home/ubuntu/cloudfs/saved_models/__home__ubuntu__cloudfs__huggingfacecache__bloom-1b7_1670626768_e2_step6017_loss1.1577339172363281'
    MODEL_NAME = '/home/ubuntu/cloudfs/saved_models/__home__ubuntu__cloudfs__huggingfacecache__bloom-1b7_1670642874_e2_step6017_loss0.9104998707771301'
    MODEL_NAME = '/home/ubuntu/cloudfs_nfs/saved_models/deep_speed_experiments/bloomz/combined_highinter_template_suspence/' \
                 'ckpt_1672379305/model-epoch=00-train_loss=1.1602.ckpt/checkpoint/mp_rank_00_model_states.pt'

    state_dict = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
    model.load_state_dict({k[len('module.model.'):]: v for k, v in state_dict['module'].items()})
    del state_dict
    torch.cuda.empty_cache()

    content = '将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入'
    title = '将数据转换成模型训练的输入'

    postfix_prompted_content = f"]\n最适合的标题类型：构造悬念（强调结果，保留部分信息，引发好奇）。\n小红书标题：[{content}]"
    prompted_content = f"根据指定内容，撰写爆款小红书笔记标题。。\n需要起标题的内容：[{title}"

    postfix_input_ids = tokenizer.encode(postfix_prompted_content)

    prefix_input_ids = tokenizer.encode(prompted_content, truncation=True,
                                             max_length=100 - len(postfix_input_ids))

    input_ids = torch.tensor([prefix_input_ids + postfix_input_ids])
    mask = torch.tensor([[0] * len(prefix_input_ids) + [1] * len(postfix_input_ids)])

    out = logits_labels_mask_to_loss(torch.tensor([[[0.5] * 500000] * (len(prefix_input_ids)+len(postfix_input_ids))]), input_ids, mask, verbose=True)

if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("Summary Task")
    total_parser.add_argument(
        '--run_ts', default=None, type=str)
    total_parser.add_argument(
        '--test_only', action='store_true', default=False)
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

    if args.test_only:
        test()
    else:

        main(args)