from operator import add
from typing import Set, Tuple, Union

import pandas as pd
from scipy.optimize import linear_sum_assignment

from tatqa_utils import *


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation):
    '''
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    '''
    answer_type = qa_annotation['answer_type'] if 'answer_type' in qa_annotation else ""
    scale = qa_annotation['scale'] if 'scale' in qa_annotation else ""
    answer_content = qa_annotation['answer']
    gold_answers = []
    if answer_type in ['multi-span', 'span']:  # list
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ['count']:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    try:
        sorted_ans = sorted(answers, key=lambda x:str(x).lower())
        ans_temp = []
        for ans in sorted_ans:
            ans_str = str(ans).lower()
            if is_number(ans_str):
                ans_num = to_number(ans_str)
                if ans_num is None:
                    if scale:
                        ans_str = ans_str + " " + str(scale)
                else:
                    if '%' in ans_str:  # has been handled the answer itself is a percentage
                        ans_str = '%.4f' % ans_num
                    else:
                        ans_str = '%.4f' % (round(ans_num, 2) * scale_to_num(scale))
            else:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            ans_temp.append(ans_str)
        return [" ".join(ans_temp)]
    except Exception as e:
        print(f'get answer str error:{e}')

    return [""]


# handle percentage
def add_percent_pred(prediction_strings, pred, pred_scale):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    if pred_scale == 'percent' and '%' not in pred_str and is_number(pred_str):  # mode only or no pred_scale num only
        pred_num = to_number(pred_str)
        if pred_num is None:
            return prediction_strings
        prediction_strings.append('%.4f' % pred_num) 
    return prediction_strings


class TATEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self, mode=None) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._head_em = 0.0
        self._node_precision = 0.0
        self._node_recall = 0.0
        self._node_f1 = 0.0
        self.head_correct_count = {"SPAN-TEXT": 0, "SPAN-TABLE": 0, "MULTI_SPAN": 0, "COUNT": 0, "ARITHMETIC": 0}
        self.head_total_count = {"SPAN-TEXT": 0, "SPAN-TABLE": 0, "MULTI_SPAN": 0, "COUNT": 0, "ARITHMETIC": 0}
        self.scale_correct_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self.scale_total_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self._count = 0
        self._details = []

    def __call__(self, ground_truth: dict,
                 prediction: Union[str, List, float, int, None],
                 pred_scale="",
                 pred_span=None,
                 gold_span=None,
                 pred_head=None,
                 gold_head=None,
                 node_precision=0,
                 node_recall=0):  # type: ignore
        """
        Parameters
        ----------
        ground_truths: ``dict``
            All the ground truth answer annotations.
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        pred_scale: ``str``
        """

        if pred_head is not None:
            if pred_head == gold_head:
                # self.head_correct_count[pred_head] += 1
                self._head_em += 1
            # self.head_total_count[gold_head] += 1

        if 'scale' in ground_truth:
            self.scale_total_count[ground_truth["scale"]] += 1
            if pred_scale == ground_truth["scale"]:
                # self.scale_correct_count[pred_scale] += 1
                self._scale_em += 1

        if prediction is None or prediction == '':
            exact_match = 0
            f1_score = 0
            span_exact_match = 0
            span_f1_score = 0
        else:
            gold_type, gold_answer, gold_scale = extract_gold_answers(ground_truth)
            if gold_answer is None or gold_answer == '':
                exact_match = 0
                f1_score = 0
                span_exact_match = 0
                span_f1_score = 0
            else:
                ground_truth_answer_strings = get_answer_str(gold_answer, gold_scale)
                prediction = prediction if isinstance(prediction, list) else [prediction]
                prediction_strings = get_answer_str(prediction, pred_scale)
                
                exact_match, f1_score = metric_max_over_ground_truths(
                    get_metrics,
                    prediction_strings,
                    ground_truth_answer_strings
                )

                if gold_type in ['arithmetic', 'count']:
                    """if gold type equals with arithmetic and count, set the f1_score == exact_match"""
                    """
                    if gold_type == 'arithmetic':
                        try:
                            if isinstance(ground_truth['answer'], list):
                                gold_ans = eval(ground_truth['answer'][0])
                            else:
                                gold_ans = ground_truth['answer']
                            if abs(pred_answer - gold_ans) < 1e-2:
                                exact_match = 1.0
                        except TypeError:
                            pass
                        except SyntaxError:
                            pass
                    """
                    f1_score = exact_match
                if not pred_span:
                    span_exact_match = 0
                    span_f1_score = 0
                else:
                    pred_span_strings = get_answer_str(pred_span, "")
                    gold_span_strings = get_answer_str(gold_span, "")
                    span_exact_match, span_f1_score = metric_max_over_ground_truths(
                        get_metrics,
                        pred_span_strings,
                        gold_span_strings,
                    )
        self._node_precision += node_precision
        self._node_recall += node_recall
        node_f1 = 0 if node_precision + node_recall == 0 else 2 * node_precision * node_recall / (node_precision + node_recall)
        self._node_f1 += node_f1
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        it = {**ground_truth,
              **{"pred": prediction,
                 "pred_scale": pred_scale,
                 "pred_head": pred_head,
                 "gold_head": gold_head,
                 "em": exact_match,
                 "f1": f1_score,
                 "pred_span": pred_span,
                 "gold_span": gold_span,
                 "span_em": span_exact_match,
                 "span_f1": span_f1_score,
                 "node_precision": node_precision,
                 "node_recall": node_recall,
                 "node_f1": node_f1
                 }}
        self._details.append(it)

    def get_overall_metric(self, reset: bool = False) -> Tuple[float, float, float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        node_precision = self._node_precision / self._count if self._count > 0 else 0
        node_recall = self._node_recall / self._count if self._count > 0 else 0
        node_f1 = self._node_f1 / self._count if self._count > 0 else 0
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        scale_score = self._scale_em / self._count if self._count > 0 else 0
        head_score = self._head_em / self._count if self._count > 0 else 0
        head_em_detail = {"SPAN-TEXT": 0, "SPAN-TABLE": 0, "MULTI_SPAN": 0, "COUNT": 0, "ARITHMETIC": 0}
        scale_em_detail = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        for k in head_em_detail.keys():
            head_em_detail[k] = self.head_correct_count[k] / self.head_total_count[k] if self.head_total_count[
                                                                                        k] > 0 else 0
        
        for k in scale_em_detail.keys():
            scale_em_detail[k] = self.scale_correct_count[k] / self.scale_total_count[k] if self.scale_total_count[
                                                                                                k] > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score, scale_score, head_score, node_precision, node_recall, node_f1

    def get_detail_metric(self):
        df = pd.DataFrame(self._details)
        if len(self._details) == 0:
            return None, None
        em_pivot_tab = df.pivot_table(index='answer_type', values=['em'],
                                       aggfunc='mean').fillna(0)

        f1_pivot_tab = df.pivot_table(index='answer_type', values=['f1'],
                                       aggfunc='mean').fillna(0)
        
        em_answer_type_tab = df.groupby('answer_type')['em'].mean()
        
        f1_answer_type_tab = df.groupby('answer_type')['f1'].mean()
        
        return em_pivot_tab, f1_pivot_tab, em_answer_type_tab, f1_answer_type_tab

    def get_raw_pivot_table(self):
        df = pd.DataFrame(self._details)
        pivot_tab = df.pivot_table(index='answer_type', values=['em'],
                                    aggfunc='count').fillna(0)
        return pivot_tab

    def get_raw(self):
        return self._details

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._head_em = 0.0
        self._node_precision = 0.0
        self._node_recall = 0.0
        self._node_f1 = 0.0
        self._count = 0
        self._details = []
        self.head_correct_count = {"SPAN-TEXT": 0, "SPAN-TABLE": 0, "MULTI_SPAN": 0, "COUNT": 0, "ARITHMETIC": 0}
        self.head_total_count = {"SPAN-TEXT": 0, "SPAN-TABLE": 0, "MULTI_SPAN": 0, "COUNT": 0, "ARITHMETIC": 0}
        self.scale_correct_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}
        self.scale_total_count = {"": 0, "thousand": 0, "million": 0, "billion": 0, "percent": 0}

    def __str__(self):
        return f"TATEmAndF1(em={self._total_em}, f1={self._total_f1}, count={self._count})"


