from datasets import load_dataset, load_from_disk
from typing import List, Union, Optional
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import random

class CustomDataset:
    def __init__(self, path: str = None, load_option: str = None):
        self.path = path
        self.load_option = load_option
        
        if self.path == "abisee/cnn_dailymail":
            self.dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
        elif self.path == "openai/gsm8k":
            self.dataset = load_dataset("openai/gsm8k", "main")
        elif self.path == "EleutherAI/hendrycks_math":
            self.dataset = None
        elif self.path == "deepmind/aqua_rat":
            self.dataset = load_dataset("deepmind/aqua_rat", "raw")
        elif self.path == 'open-r1/OpenR1-Math-220k':
            self.dataset = load_dataset("open-r1/OpenR1-Math-220k", "default")
        elif self.path == 'qiaojin/PubMedQA':
            self.dataset = None
        elif self.path == 'hotpotqa/hotpot_qa':
            self.dataset = load_dataset('hotpotqa/hotpot_qa', 'fullwiki')
        elif self.path == 'allenai/openbookqa':
            self.dataset = load_dataset('allenai/openbookqa', 'additional')
        elif self.load_option == 'disk':
            if self.path == '/project/phan/nbn4/datasets':
                self.dataset_phase1 = load_from_disk(f"{self.path}/Phase1")['train']
                self.dataset_phase2 = load_from_disk(f"{self.path}/Phase2")
                self.dataset_phase3 = load_from_disk(f"{self.path}/Phase3")['train']
                self.dataset = concatenate_datasets(
                    [self.dataset_phase1,
                    self.dataset_phase2,
                    self.dataset_phase3]
                )
            elif self.path in ['/project/phan/nbn4/datasets/Phase1', '/project/phan/nbn4/datasets/Phase2', '/project/phan/nbn4/datasets/Phase3']:
                self.dataset = load_from_disk(self.path)
        else:
            self.dataset = load_dataset(self.path)

    def split_dataset(self):
        if self.path == 'EleutherAI/hendrycks_math':
            train_dataset = concatenate_datasets(
                [load_dataset("EleutherAI/hendrycks_math", "algebra", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "counting_and_probability", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "geometry", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "intermediate_algebra", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "number_theory", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "prealgebra", split='train'),
                 load_dataset("EleutherAI/hendrycks_math", "precalculus", split='train')]
            )
            val_dataset = concatenate_datasets(
                [load_dataset("EleutherAI/hendrycks_math", "algebra", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "counting_and_probability", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "geometry", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "intermediate_algebra", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "number_theory", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "prealgebra", split='test'),
                 load_dataset("EleutherAI/hendrycks_math", "precalculus", split='test')]
            )
            test_dataset = None
        elif self.path == 'qiaojin/PubMedQA':
            train_dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")['train']
            val_dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")['train']
            test_dataset = None
        else:
            if 'train' in self.dataset:
                train_dataset = self.dataset['train']
            else:
                train_dataset = self.dataset

            if 'validation' in self.dataset:
                val_dataset = self.dataset['validation']
            else:
                val_dataset = None

            if 'test' in self.dataset:
                test_dataset = self.dataset['test']
            else:
                test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def format_aquarat(self, batch, mode):
        questions = []
        answers = []
        for q, o, r, c in zip(batch['question'], batch['options'], batch['rationale'], batch['correct']):
            formatted_options = "\n".join(o)
            if mode == 'train':
                question = f"{q}\nWhich of the following options is correct?\n{formatted_options}"
            elif mode == 'inference':
                question = "Let's think step by step and write only the choice letter in the format '#### {answer}'\n" + f"{q}\nWhich of the following options is correct?\n{formatted_options}"
            answer = f"{r}\n#### {c}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch

    def format_openr1(self, batch, mode):
        questions = []
        answers = []
        for p, s, a in zip(batch['problem'], batch['solution'], batch['answer']):
            if mode == 'train':
                question = p
            elif mode == 'inference':
                question = "Let's think step by step and write your answer at the end in the format '#### {answer}'\n" + p
            answer = f"{s}\n#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch

    def format_gsm8k(self, batch, mode):
        questions = []
        answers = []
        for q, a in zip(batch['question'], batch['answer']):
            if mode == 'train':
                question = q
            elif mode == 'inference':
                question = "Let's think step by step and write your answer at the end in the format '#### {answer}'\n" + q
            answer = a
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch

    def format_mathqa(self, batch, mode):
        questions = []
        answers = []
        for p, r, o, c in zip(batch['Problem'], batch['Rationale'], batch['options'], batch['correct']):
            if mode == 'train':
                question = f"{p}\n{o}"
            elif mode == 'inference':
                question = "\nLet's think step by step and write only the choice letter at the end in the format '#### {answer}'\n" + f"{p}\n{o}"
            r = r.replace('"', '')
            answer = f"{str(r)}\n#### {c}"
            batch['question'] = question
            batch['answer'] = answer
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        return batch

    def format_svamp(self, batch, mode):
        answers = []
        questions = []
        for q, e, a in zip(batch['question_concat'], batch['Equation'], batch['Answer']):
            if mode == 'train':
                question = q
            elif mode == 'inference':
                question = "\nLet's think step by step and write your answer at the end in the format '#### {answer}'\n" + q
            answer = f"{e}\n#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_strategyqa(self, batch, mode):
        questions = []
        answers = []
        for q, f, a in zip(batch['question'], batch['facts'], batch['answer']):
            if mode == 'train':
                question = f"{f}\n{q}"
            elif mode == 'inference':
                question = "Answer only 'True' or 'False' in the format '#### {answer}'.\n" + f + "\n" + q
            answer = f"#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_commonsenseqa(self, batch, mode):
        questions = []
        answers = []
        for a, q in zip(batch['answerKey'], batch['question_concat']):
            if mode == 'train':
                question = q
            elif mode == 'inference':
                question = "Answer only the choice letter in capital letter in the format: '#### {answer}'.\n" + q
            answer = f"#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch

    def format_dialogsum(self, batch):
        questions = []
        answers = []
        for d, s in zip(batch['dialogue'], batch['summary']):
            question = f"Summarize the following dialogue.\n{d}"
            answer = f"{s}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch

    def format_samsum(self, batch):
        questions = []
        answers = []
        for d, s in zip(batch['dialogue'], batch['summary']):
            question = f"Summarize the following dialogue.\n{d}"
            answer = f"{s}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']

        return batch

    def format_openbookqa(self, batch, mode):
        questions = []
        answers = []
        for q, c, f, a in zip(batch['question_stem'], batch['choices'], batch['fact1'], batch['answerKey']):
            c = c['text']
            formatted_options = f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}"
            if mode == 'train':
                question = f"{f}.\n{q}?\nWhich of the following options is correct?\n{formatted_options}"
            elif mode == 'inference':
                question = "Let's think step by step and write only the choice letter in the format '#### {answer}'\n" + f"{f}.\n{q}?\nWhich of the following options is correct?\n{formatted_options}"
            answer = f"#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_pubmedqa(self, batch, mode):
        questions = []
        answers = []
        for q, c, f in zip(batch['question'], batch['context'], batch['final_decision']):
            c = '\n'.join(c['contexts'])
            if mode == 'train':
                question = f"{c}\n{q}"
            elif mode == 'inference':
                question = "Let's think step by step and write the answer 'yes', 'no', or 'maybe' in the format '#### {answer}'\n" + f"{c}\n{q}"
            answer = f"#### {f}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_hotpotqa(self, batch, mode):
        questions = []
        answers = []
        for q, c, a in zip(batch['question'], batch['context'], batch['answer']):
            c = '\n'.join(['\n'.join(s) for s in c['sentences']])
            if mode == 'train':
                question = f"{c}\n{q}"
            elif mode == 'inference':
                question = "Let's think step by step and write the answer in the format '#### {answer}'\n" + f"{c}\n{q}"
            answer = f"#### {a}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_drop(self, batch, mode):
        questions = []
        answers = []
        for q, p, a in zip(batch['question'], batch['passage'], batch['answers_spans']):
            if mode == 'train':
                question = f"{p}\n{q}"
            elif mode == 'inference':
                question = "Let's think step by step and write the answer in the format '#### {answer}'\n" + f"{p}\n{q}"
            answer = f"#### {a['spans'][0]}"
            questions.append(question)
            answers.append(answer)

        batch['question'] = questions
        batch['answer'] = answers

        if 'predicted_output' in batch:
            print(f"Actual labels: {batch['answer'][:5]}")
            print(f"Using small model labels: {batch['predicted_output'][:5]}")
            batch['answer'] = batch['predicted_output']
        return batch
    
    def format_squad(self, batch, mode):
        questions = []
        answers = []
        for c, q, a in zip(batch['context'], batch['question'], batch['answers']):
            if mode == 'train':
                question = f"{c}\n{q}"
            elif mode == 'inference':
                question = "Let's think step by step and write the answer in the format '#### {answer}'\n" + f"{c}\n{q}"
            answer = f"#### {a['text'][0]}"
            questions.append(question)
            answers.append(answer)
        
        batch['question'] = questions
        batch['answer'] = answers
        return batch

    def format_mass(self, batch):
        questions = []
        answers = []
        for i, o in zip(batch['instruction'], batch['output']):
            i = i.replace("Carefully think through the problem and provide a complete and efficient solution in an appropriate format that addresses the following task:\n### Instruction:\n", "")
            i = i.replace("\n\n### Response:", "")
            o = o.split("<think>\n")[-1]
            o = o.replace("</think>\n", "")
            question = i
            answer = o
            questions.append(question)
            answers.append(answer)
        
        batch['instruction'] = questions
        batch['output'] = answers
        return batch

    def preprocess_function_causal(self,
                                    examples: dict = None,
                                    input_key: str = None,
                                    target_key: str = None,
                                    prefix: str = '',
                                    postfix: str = '',
                                    log: bool = False,
                                    max_length: int = None,
                                    padding: Union[str, bool] = None,
                                    truncation: bool = None,
                                    tokenizer: AutoTokenizer = None,
                                    mode: str = None,
                                    dataset: str = None) -> dict:
        if mode == "train":
            inputs = [prefix + str(doc) + postfix for doc in examples[input_key]]
            targets = examples[target_key]
            full_texts = [inp + tgt + "<|im_end|>" for inp, tgt in zip(inputs, targets)]
            
            if log:
                print(f"===EXAMPLE SAMPLE (TRAINING)===\n{full_texts[0]}")

            model_inputs = tokenizer(
                full_texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
        elif mode == 'inference':
            inputs = [prefix + str(doc) + postfix for doc in examples[input_key]]
            targets = examples[target_key]
            input_texts = [inp for inp in inputs]
            target_texts = [tgt for tgt in targets]

            if log:
                print(f"===EXAMPLE SAMPLE (EVAL)===\nINPUT:\n{input_texts[0]}\n==========\nOUTPUT:\n{target_texts[0]}")

            model_inputs = tokenizer(
                input_texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )

            labels = tokenizer(
                target_texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )

            model_inputs["labels"] = labels["input_ids"]

        return model_inputs

def random_sample(dataset, n_samples: int = None, seed: int = 42) -> List[dict]:
    random.seed(seed)
    rand_idx = random.sample(list(range(len(dataset))), n_samples)
    subset = dataset.select(rand_idx)
    return subset