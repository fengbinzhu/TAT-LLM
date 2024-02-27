#!/usr/bin/python
import argparse
import json
from tatqa_utils import extract_all_nums_from_str
from tatqa_metric import *
from typing import Any, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

dataset = 'tatdqa'
model = 'tat-llm-all-7b'
mode = 'sft'
with_scale = True
out_dir = Path('./process/prediction/')

def measure_match(ans_num, pred_num):
    if ans_num is None or pred_num is None:
        return False
    if str(ans_num) in ['true', 'false']:
        if ans_num in pred_num:
            return True
        return False
    if len(str(pred_num)) > 20:
        return False
    gap = min(abs(abs(ans_num) - abs(pred_num)),  abs(abs(ans_num) - abs(pred_num) * 100), abs(abs(ans_num) - abs(pred_num) / 100))

    if gap == 0:
        return True

    # Tolerate 1% difference
    if gap < 1 and abs(ans_num) > 100:
        return True
    if gap < 0.1 and abs(ans_num) > 10:
        return True
    if gap <= 0.01:
        if abs(ans_num) > 1:
            return True
        else:
            if gap < 0.001:
                return True
            elif ans_num != 0 and gap / abs(ans_num) < 0.01:
                return True
    return False


def analyze_sft_response(llm_response_text):
    arr = llm_response_text.split('\n')
    res_rows = [x for x in arr if '|' in x]
    res_map = {}
    for row in res_rows:
        cols = [x.strip() for x in row.split('|') if x.strip() != '']
        k = cols[0]
        v = cols[1]
        res_map[k] = v
    return res_map

def clean_equation(equation):
    import string
    res = ''
    for c in equation:
        if c not in string.ascii_lowercase and c not in ['&','%', ',', '$']:
            res += c
    return res

def parse_pred_answer(gold_qa, llm_response_text, dataset='finqa'):
    preds = []
    pred_scale = ''
    gold_type, gold_answers, gold_scale =  extract_gold_answers(gold_qa)
    llm_response_text = llm_response_text.lower()

    llm_ans_str = llm_response_text.strip()
    pred_scale = ''
    if 'the answer is:' in llm_response_text:
        llm_ans_str = llm_response_text.split('the answer is:')[1].strip().replace('</s>', '')
        if dataset in ['tatqa', 'wikitq']:
            arr = llm_ans_str.split('####')
            llm_ans_str = arr[0].strip()
            if len(arr) > 1:
                pred_scale = arr[1].replace('and its corresponding scale is:', '').strip()
                pred_scale = '' if pred_scale == 'none' else pred_scale
            else:
                if gold_scale != '' and gold_scale in llm_ans_str:
                    pred_scale = gold_scale

    res_map = ''
    if '4' in res_map:
        pred_scale = res_map['4'].strip()
        pred_scale = '' if pred_scale == 'none' else pred_scale

    try:
        # External Executor
        res_map = analyze_sft_response(llm_response_text)
        if dataset == 'finqa':
            if res_map['3'] in ['true', 'yes']:
               llm_ans_str = 'true'
            elif res_map['3'] in ['false', 'no']:
               llm_ans_str = 'false'
            else:
                equation = clean_equation(res_map['2'])
                llm_ans_str = str(round(eval(equation), 4))
        elif dataset in ['tatqa', 'wikitq']:
            if res_map['1'] == 'arithmetic' and '3' in res_map:
                equation = clean_equation(res_map['3'])
                llm_ans_str = str(round(eval(equation), 4))
            if res_map['1'] == 'count' and '2' in res_map:
                  evidencs = res_map['2'].strip()
                  llm_ans_str = len(evidencs.split('#'))
            if res_map['1'] == 'multiple spans' and '2' in res_map:
                  llm_ans_str = res_map['2'].strip()
            if res_map['1'] == 'single span':
                  llm_ans_str = res_map['2'].strip()
    except Exception as e:
        print(f'equation error:{e}')
        pass

    flag = 0
    for gold_answer in gold_answers:
        if dataset == 'wikitq':
            preds = llm_ans_str

        if gold_type in ['count', 'arithmetic']:
            gold_answer = gold_answer.lower()
            if gold_answer in ['true', 'false'] and measure_match(gold_answer, llm_ans_str):
                preds.append(gold_answer)
                flag = 1
                break
            gold_answer_num = to_number(gold_answer)
            nums = extract_all_nums_from_str(llm_ans_str)
            nums.reverse()
            found = False
            for n in nums:
                if measure_match(gold_answer_num, n):
                    found = True
                    flag = 1
                    preds.append(gold_answer_num)
                    break
            if not found:
                preds = nums
        elif gold_type in ['span']:
            if mode == 'infer' and gold_answer in llm_ans_str:
                preds.append(gold_answer)
                pred_scale = gold_scale
            else:
                preds.append(llm_ans_str)
        elif gold_type in ['multi-span']:
            if mode == 'sft':
                preds = llm_ans_str.split('#')
            else:
               if gold_answer in llm_ans_str:
                    preds.append(gold_answer)
    pred_str = ""

    # Ignore scale for zero-shot inference
    if not with_scale:
        pred_scale = gold_scale
    return preds, pred_scale, pred_str, flag


def evaluate_json(golden_answers: Dict[str, Any], llm_predictions: Dict[str, Any]) -> Tuple[float, float]:

    em_and_f1 = TATEmAndF1()
    for qas in tqdm(golden_answers):
        if "questions" in qas:
            for qa in qas["questions"]:
                query_id = qa["uid"]
                pred_answer, pred_scale = None, None
                if query_id in llm_predictions:
                    llm_response = llm_predictions[query_id]
                    if isinstance(llm_response, str):
                        llm_response_text = llm_response
                        pred_answer, pred_scale, pred_str, flag = parse_pred_answer(qa, llm_response_text, 'tatqa')
                    else:
                        for llm_response_text in llm_response:
                            pred_answer, pred_scale, pred_str, flag = parse_pred_answer(qa, llm_response_text, 'tatqa')
                            if flag:
                                break
                    pred_str
                em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)
        else:
            #finqa & wikitq
            pred_answer, pred_scale = None, None
            qa = qas
            query_id = qa["id"]
            if query_id in llm_predictions:
                llm_response = llm_predictions[query_id]
                if isinstance(llm_response, str):
                    llm_response_text = llm_response
                    pred_answer, pred_scale, pred_str, flag = parse_pred_answer(qa, llm_response_text, dataset)
                else:
                    for llm_response_text in llm_response:
                        pred_answer, pred_scale, pred_str, flag = parse_pred_answer(qa, llm_response_text, dataset)
                        if flag:
                            break
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    global_em, global_f1, global_scale, _, _, _, _ = em_and_f1.get_overall_metric()
    print("----")
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("Scale score {0:.2f}".format(global_scale * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")

    detail_raw = em_and_f1.get_raw_pivot_table()
    print("---- raw detail ---")
    print(detail_raw)
    em_pivot_tab, f1_pivot_tab, em_answer_type_tab, f1_answer_type_tab = em_and_f1.get_detail_metric()
    print("---- em detail ---")
    print(em_pivot_tab)
    print("---- f1 detail ---")
    print(f1_pivot_tab)


def evaluate_prediction_file(gold_path: str,
                             pred_path: str):
    golden_answers = json.load(open(gold_path, encoding='utf-8'))
    llm_predictions = json.load(open(pred_path, encoding='utf-8'))
    llm_predictions = {one['id']:one['prediction'] for one in llm_predictions}
    evaluate_json(golden_answers, llm_predictions)


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='evaluation of TAT-LLM')


    parser.add_argument("--dataset_name",
                        type=str,
                        required=False,
                        default="finqa",
                        help='The dataset name must be given')
    parser.add_argument("--model_type",
                        type=str,
                        required=False,
                        default= "fft",
                        help='The model type which is either fft or lora')
    parser.add_argument("--model_name",
                        type=str,
                        required=False,
                        default= "tat-llm-7b",
                        help='The path of the prediction file')

    args = parser.parse_args()

    dataset = args.dataset_name
    model_type = args.model_type
    model = args.model_name

    gold_path = f"./data/original/{dataset}/{dataset}_dataset_test.json"
    pred_path = f"./data/prediction/{model}/{model_type}/{dataset}_{model.replace('-','_')}_pred.json"

    # hash()
    evaluate_prediction_file(gold_path, pred_path)

