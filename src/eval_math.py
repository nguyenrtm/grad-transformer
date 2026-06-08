import math
from grader import math_equal
import wandb
import re

def _last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if left_brace_idx is None or right_brace_idx is None:
        return None
    return string[left_brace_idx + 1: right_brace_idx].strip()

def evaluate_math_accuracy(predictions, references, table=True):
    correct = 0
    total = len(predictions)
    i = 0
    records_table = None
    if table == True:
        records_table = wandb.Table(columns=["Prediction", "Ground Truth"])
    for pred, ref in zip(predictions, references):
        i += 1
        pred_ans = _last_boxed_only_string(pred)
        ref_ans = _last_boxed_only_string(ref)
        if not ref_ans:
            print(f"ERROR IN REFERENCE:\n{ref}")
            total -= 1
            continue
        elif not pred_ans:
            pred_ans = '[DID NOT PREDICT]'
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | INCORRECT")
            if table == True:
                records_table.add_data(pred, ref)
            continue
        try:
            if r'\pi' in pred_ans or r'\pi' in ref_ans:
                equiv = any(
                    math_equal(pred_ans, ref_ans, timeout=True, pi=val)
                    for val in [math.pi, 3.14]
                )
            else:
                equiv = math_equal(pred_ans, ref_ans, timeout=True)
        except (ValueError, TypeError):
            equiv = False
        if equiv:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | CORRECT")
            correct += 1
        else:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | INCORRECT")
        if table == True:
            records_table.add_data(pred, ref)
    print(f"Correctly predicted: {correct}/{total}")
    return correct / total if total > 0 else 0.0, records_table

def evaluate_math_gsm8k_accuracy(predictions, references, table=True, extract_integer=False):
    correct = 0
    total = len(predictions)
    i = 0
    records_table = None
    def extract_integer(pred_str):
        match = re.search(r"\d+", pred_str)
        return int(match.group()) if match else None
    if table == True:
        records_table = wandb.Table(columns=["Prediction", "Ground Truth"])
    for pred, ref in zip(predictions, references):
        i += 1
        pred_ans = _last_boxed_only_string(pred)
        ref_ans = ref.split('####')[1].strip()
        if not ref_ans:
            print(f"ERROR IN REFERENCE!")
            total -= 1
            continue
        elif not pred_ans:
            pred_ans = '[DID NOT PREDICT]'
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | INCORRECT")
            if table == True:
                records_table.add_data(pred, ref)
            continue
        try:
            if extract_integer:
                pred_ans = str(extract_integer(pred_ans))
            if r'\pi' in pred_ans or r'\pi' in ref_ans:
                equiv = any(
                    math_equal(pred_ans, ref_ans, timeout=True, pi=val)
                    for val in [math.pi, 3.14]
                )
            else:
                equiv = math_equal(pred_ans, ref_ans, timeout=True)
        except (ValueError, TypeError):
            equiv = False
        if equiv:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | CORRECT")
            correct += 1
        else:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<40} | Ground Truth: {ref_ans:<40} | INCORRECT")
        if table == True:
            records_table.add_data(pred, ref)
    print(f"Correctly predicted: {correct}/{total}")
    return correct / total if total > 0 else 0.0, records_table

def evaluate_math_reasoning_accuracy(
        predictions, 
        references, 
        ans_template,
        ref_template,
        table=True
    ):
    correct = 0
    total = len(predictions)
    i = 0
    records_table = None
    if table == True:
        records_table = wandb.Table(columns=["Prediction", "Ground Truth"])
    for pred, ref in zip(predictions, references):
        # print(f"*************\n{pred}\n====\n{ref}\n")
        i += 1
        try:
            pred_ans = pred.split(ans_template)[-1].strip()
            pred_ans = re.sub(r'\.$', '', pred_ans)
        except:
            continue
        ref_ans = None
        try:
            ref_ans = ref.split(ref_template)[-1].strip()
        except:
            continue
        if ref_ans == None:
            print(f"ERROR IN REFERENCE:\n{ref}")
            total -= 1
            continue
        elif not pred_ans:
            pred_ans = '[DID NOT PREDICT]'
            if table == True:
                print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | INCORRECT")
                records_table.add_data(pred, ref)
            continue
        try:
            if r'\pi' in pred_ans or r'\pi' in ref_ans:
                equiv = any(
                    math_equal(pred_ans, ref_ans, timeout=True, pi=val)
                    for val in [math.pi, 3.14]
                )
            else:
                equiv = math_equal(pred_ans, ref_ans, timeout=True)
        except (ValueError, TypeError, AttributeError):
            equiv = False
        if equiv:
            if table == True:
                print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | CORRECT")
            correct += 1
        else:
            if table == True:
                print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | INCORRECT")
        if table == True:
            records_table.add_data(pred, ref)
    print(f"Correctly predicted: {correct}/{total}")
    return correct / total if total > 0 else 0.0, records_table

def evaluate_commonsenseqa_accuracy(
        predictions, 
        references, 
        ans_template,
        ref_template,
        table=True
    ):
    correct = 0
    total = len(predictions)
    i = 0
    records_table = None
    if table == True:
        records_table = wandb.Table(columns=["Prediction", "Ground Truth"])
    for pred, ref in zip(predictions, references):
        i += 1
        try:
            pred_ans = pred.split(ans_template)[-1].strip()
            pred_ans = re.sub(r'\.$', '', pred_ans).replace('(', '')
            pred_ans = pred_ans.split(')')[0]
        except:
            continue
        ref_ans = None
        try:
            ref_ans = ref.split(ref_template)[-1].strip()
        except:
            continue
        if ref_ans == None:
            print(f"ERROR IN REFERENCE:\n{ref}")
            total -= 1
            continue
        elif not pred_ans:
            pred_ans = '[DID NOT PREDICT]'
            print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | INCORRECT")
            if table == True:
                records_table.add_data(pred, ref)
            continue
        try:
            if r'\pi' in pred_ans or r'\pi' in ref_ans:
                equiv = any(
                    math_equal(pred_ans, ref_ans, timeout=True, pi=val)
                    for val in [math.pi, 3.14]
                )
            else:
                equiv = math_equal(pred_ans, ref_ans, timeout=True)
        except (ValueError, TypeError):
            equiv = False
        if equiv:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | CORRECT")
            correct += 1
        else:
            print(f"Sample {i:<4} | Predicted: {pred_ans:<20} | Ground Truth: {ref_ans:<20} | INCORRECT")
        if table == True:
            records_table.add_data(pred, ref)
    print(f"Correctly predicted: {correct}/{total}")
    return correct / total if total > 0 else 0.0, records_table