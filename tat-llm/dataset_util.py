
from collections import defaultdict
from multiprocessing.sharedctypes import Value

import numpy as np

import torch
np.set_printoptions(threshold=np.inf)
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .file_utils import is_scatter_available
from .data_util import *
from .data_tools import *
from .tag_constant import *
import copy

# soft dependency
if is_scatter_available():
    pass

def convert_start_end_tags(split_tags, paragraph_index):
    in_split_tags = split_tags.copy()
    split_tags = [0 for i in range(len(split_tags))]
    for i in range(len(in_split_tags)):
        if in_split_tags[i] == 1:
            current_index = paragraph_index[i]
            split_tags[i] = 1
            paragraph_index_ = paragraph_index[i:]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[j] == current_index:
                    split_tags[i + j] = 1
                else:
                    break
            break
    for i in range(1, len(in_split_tags)):
        if in_split_tags[-i] == 1:
            current_index = paragraph_index[-i]
            split_tags[-i] = 1
            paragraph_index_ = paragraph_index[:-i]
            for j in range(1, len(paragraph_index_)):
                if paragraph_index_[-j] == current_index:
                    split_tags[-i - j] = 1
                else:
                    break
            break
    del in_split_tags
    return split_tags


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def sortFunc(elem):
    return elem[1]


def get_order_by_tf_idf(question, paragraphs):
    sorted_order = []
    corpus = [question]
    for order, text in paragraphs.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]


def get_mapping_pos_block(block_map, mapping):
    block_nums = []
    block_pos = []
    # pattern = re.compile("\d+\.\d+%?|\d+%?")
    # neg_pat = re.compile("\((\d+\.\d+%?|\d+%?)\)")
    for one in mapping:
        for block_id in one.keys():
            index = one[block_id]
            if block_id not in block_map.keys():
                continue
                # raise ValueError(f'cannot find {block_id} in {block_map.keys()} ')
            seg = block_map[block_id]['text'][index[0]:index[1]].strip().split()
            begin = index[0]
            for s in seg:
                if is_number(s):
                    block_nums.append(str(to_number(s)))
                    block_pos.append((block_id, begin, begin + len(s)))
                begin += len(s)
    return block_nums, block_pos


def get_num_pos_block_outseq(block_map, mapping, derivation):
    derivation = change2neg(normalize_derivation(derivation))
    nums = get_nums_from_derivation(derivation)
    nums_neg = [n for n in nums if n[0] == '-']
    out = from_infix_to_prefix(dele(seg(derivation, nums, nums_neg)))

    if not istree(out):
        derivation = proderivation(derivation)
        nums = get_nums_from_derivation(derivation)
        nums_neg = [n for n in nums if n[0] == '-']
        out = from_infix_to_prefix(dele(seg(derivation, nums, nums_neg)))
    block_nums, block_pos = get_mapping_pos_block(block_map, mapping)

    return out, nums, block_nums, block_pos


def get_sorted_block_uuids(question, block_map):
    uuids = []
    corpus = [question]
    sorted_uuids =  []
    for uuid, b in block_map.items():
        text = b['text']
        words = text.split()
        if any(is_number(w) for w in words[-3:]):
            sorted_uuids.append(uuid)
        else:
            corpus.append(text)
            uuids.append(uuid)
            
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return sorted_uuids + [uuids[index] for index in idx]


def get_tokens_from_ids(ids, tokenizer):
    tokens = []
    sub_tokens = []
    for id in ids:
        token = tokenizer._convert_id_to_token(id)
        if len(sub_tokens) == 0:
            sub_tokens.append(token)
        elif str(token).startswith("##"):
            sub_tokens.append(token[2:])
        elif len(sub_tokens) != 0:
            tokens.append("".join(sub_tokens))
            sub_tokens = [token]
    tokens.append("".join(sub_tokens))
    return "".join(tokens)


def get_num_pos_outseq(table, paragraphs, mapping, derivation):
    derivation = change2neg(normalize_derivation(derivation))
    nums = get_nums_from_derivation(derivation)
    nums_neg = [n for n in nums if n[0] == '-']
    out = from_infix_to_prefix(dele(seg(derivation, nums, nums_neg)))

    if not istree(out):
        derivation = proderivation(derivation)
        nums = get_nums_from_derivation(derivation)
        nums_neg = [n for n in nums if n[0] == '-']
        out = from_infix_to_prefix(dele(seg(derivation, nums, nums_neg)))

    table_nums, table_pos, para_nums, para_pos = get_mapping_pos(table, paragraphs, mapping)

    return out, nums, table_nums, table_pos, para_nums, para_pos

def text_tokenize(text: str, text_tags, tokenizer):
    if not text:
        return [], []
    words = []
    word_tags = []
    prev_is_whitespace = True
    word2text_idx = []
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~","?","!", "$","(",")", "€","£"]:
            words.append(c)
            word_tags.append(text_tags[i]) # use the first char
            word2text_idx.append(i)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                words.append(c)
                word_tags.append(text_tags[i])
                word2text_idx.append(i)
            else:
                words[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    token_tags = []
    token2word_idx = []
    token2text_idx = []
    for w_idx, (word, tag, text_idx) in enumerate(list(zip(words, word_tags, word2text_idx))):
        if w_idx != 0:
            sub_tokens = tokenizer._tokenize(" " + word)
        else:
            sub_tokens = tokenizer._tokenize(word)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            token_tags.append(tag)
            token2word_idx.append(w_idx)
            token2text_idx.append(text_idx)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    assert len(split_tokens) == len(token_tags)
    return ids, split_tokens, token_tags, token2text_idx, token2word_idx, words



def text_tokenize_dqa(text: str, text_tags, tokenizer):
    if not text:
        return [], []
    words = []
    word_tags = []
    prev_is_whitespace = True
    word2text_idx = []
    index=get_index_num_date(text_tags)
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
        elif c in ["-", "–", "~","?","!", "$","(",")", "€","£"]:
            words.append(c)
            word_tags.append(text_tags[i]) # use the first char
            word2text_idx.append(i)
            prev_is_whitespace = True
        elif i in index:
            words.append(c)
            word_tags.append(text_tags[i]) # use the first char
            word2text_idx.append(i)
            prev_is_whitespace = False
        else:
            if prev_is_whitespace:
                words.append(c)
                word_tags.append(text_tags[i])
                word2text_idx.append(i)
            else:
                words[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)

    split_tokens = []
    token_tags = []
    token2word_idx = []
    token2text_idx = []
    for w_idx, (word, tag, text_idx) in enumerate(list(zip(words, word_tags, word2text_idx))):
        if w_idx != 0:
            sub_tokens = tokenizer._tokenize(" " + word)
        else:
            sub_tokens = tokenizer._tokenize(word)

        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            token_tags.append(tag)
            token2word_idx.append(w_idx)
            token2text_idx.append(text_idx)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    assert len(split_tokens) == len(token_tags)
    return ids, split_tokens, token_tags, token2text_idx, token2word_idx, words

def get_index_num_date(tags):
    index=[]
    begin=0
    flag=0
    current_number=0
    for i,m in enumerate(tags):
        if flag==0:
            if m==1 or m==2:
                index.append(i)
                current_number=m
                flag=1
        if flag==1:
            if current_number==1 and m==2:
                index.append(i-1)
                index.append(i)
                current_number=m
            elif current_number==2 and m==1:
                index.append(i-1)
                index.append(i)
                current_number=m
            elif m!=1 and m!=2:
                index.append(i)
                flag=0
    return index


def block_tokenize_dqa(word_lists, text, text_tags, bbox_list,tokenizer):
    words = []
    word_tags = []
    prev_is_whitespace = True
    word2text_idx = []
    new_bbox_lists=[]
    bbox_index=0
    index = get_index_num_date(text_tags)
    if '\t' in text:text = text.replace('\t','')
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            prev_is_whitespace = True
            bbox_index+=1
        elif c in ["-", "–", "~","?","!", "$","(",")", "€","£"]:
            words.append(c)
            word_tags.append(text_tags[i]) # use the first char
            word2text_idx.append(i)
            new_bbox_lists.append(bbox_list[bbox_index])
            prev_is_whitespace = True
        elif i in index:
            words.append(c)
            word_tags.append(text_tags[i]) # use the first char
            word2text_idx.append(i)
            new_bbox_lists.append(bbox_list[bbox_index])
            prev_is_whitespace = False
        else:
            if prev_is_whitespace:
                words.append(c)
                word_tags.append(text_tags[i])
                word2text_idx.append(i)
                new_bbox_lists.append(bbox_list[bbox_index])
            else:
                words[-1] += c
            prev_is_whitespace = False  # char_to_word_offset.append(len(tokens) - 1)
    # word_lists,bbox_list=new_word_list(word_lists,bbox_list)
    word_bboxes=[]
    block_token_bboxes=[]
    assert len(words) == len(word_tags) == len(word2text_idx) == len(new_bbox_lists)
    # word2text_idx = []
    # word2text_idx.append(0)
    # word_tags.append(text_tags[0])
    # index=0
    # for i in word_lists[:-1]:
    #     index+=len(i)+1
    #     word_tags.append(text_tags[index])
    #     word2text_idx.append(index)

    split_tokens = []
    token_tags = []
    token2word_idx = []
    token2text_idx = []
    for w_idx, (word, tag, text_idx,bbox) in enumerate(list(zip(words, word_tags, word2text_idx,new_bbox_lists))):
        if w_idx != 0:
            sub_tokens = tokenizer._tokenize(" " + word)
        else:
            sub_tokens = tokenizer._tokenize(word)
        word_bboxes.append(bbox)
        for sub_token in sub_tokens:
            split_tokens.append(sub_token)
            token_tags.append(tag)
            token2word_idx.append(w_idx)
            token2text_idx.append(text_idx)
            block_token_bboxes.append(bbox)

    ids = tokenizer.convert_tokens_to_ids(split_tokens)
    assert len(split_tokens) == len(token_tags) == len(token2word_idx) == len(token2text_idx)
    return ids, split_tokens, token_tags, token2text_idx, token2word_idx,word_bboxes,block_token_bboxes, words


def update_tag_seq_offsets(tag_offsets, mode):
    result = copy.deepcopy(tag_offsets)
    self_seq_offsets = tag_offsets[TAG_SELF[0]]['seq_offsets']
    self_start = self_seq_offsets[0]
    self_end = self_seq_offsets[1]
    for tag, tag_values in tag_offsets.items():
        if tag == TAG_SELF[0]:
            if 'self_answer_offsets' in tag_offsets[TAG_SELF[0]]:
                new_self_answer_offsets = []
                for one in tag_offsets[TAG_SELF[0]]['self_answer_offsets']:
                    new_one = list(one)
                    token_start, token_end = new_one[2]
                    one_start = self_start + token_start
                    one_end = self_start + token_end
                    if mode != 'test' and one_end > self_end:
                        raise ValueError(f' The answer offsets exceed the max len of {one_end} > {self_end}')
                    new_one[2] = [one_start, one_end]
                    new_self_answer_offsets.append(new_one)
                result[TAG_SELF[0]]['self_answer_offsets'] = new_self_answer_offsets
                # if len(new_self_answer_offsets) > 1:
                #     print(result[TAG_SELF[0]])
        else: # num + date
            new_tag_values = []
            for one in tag_values:
                is_target = one['is_target']
                token_start, token_end = one['token_offsets']
                one_start = self_start + token_start
                one_end = self_start + token_end
                # drop it when the end exceeds the max length of self
                if one_end > self_end:
                    if mode != 'test' and is_target == 1:
                        raise ValueError(f'The answer node exceed the max len of {one_end} > {self_end}')
                    else:
                        continue
                one['seq_offsets'] = [one_start, one_end]
                new_tag_values.append(one)
            result[tag] = new_tag_values
    return result

def concat_dqa(question_metas,
                block_metas,
                sep_start,
                sep_end,
                max_pieces,
                mode='train'):
    question_ids, question_tokens, question_token_tags, question_offsets, question_token2word_idx, question_words = question_metas

    input_ids = torch.zeros([1, max_pieces])
    input_segments = torch.zeros_like(input_ids)
    input_bbox_orders = torch.zeros_like(input_ids)
    input_tags = torch.zeros_like(input_ids)
    input_bboxes = torch.zeros([1, max_pieces, 4], dtype=int)
    question_mask = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    bbox_mask = torch.zeros_like(input_ids)

    # add question to the seq
    question_ids = [sep_start] + question_ids + [sep_end]
    question_token_tags = [0] + question_token_tags + [0]
    question_length = len(question_ids)
    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_tags[0, :question_length] = torch.from_numpy(np.array(question_token_tags))
    question_mask[0, 1: question_length - 1] = 1 # remove the [start] and [end ]

    question_offsets[TAG_SELF[0]]['seq_offsets'] = [1, question_length - 1]
    new_question_offsets = update_tag_seq_offsets(question_offsets, mode)

    new_question_metas = (question_ids, question_tokens, question_token_tags, new_question_offsets, question_token2word_idx, question_words)

    # add bbox to the seq
    context_length_limitation = max_pieces - 1 - question_length
    context_length = 0
    bbox_length = 0
    bbox_break=0
    new_block_metas = []
    new_word_bboxes=[]
    new_block_token_bboxes=[]

    for bbox_idx, (bbox_order, bbox_ids, bbox_tokens, bbox_token_tags, bbox_offsets, bbox_token2word_idx, bbox_words,word_bboxes,block_token_bboxes,bbox_lists) in enumerate(block_metas):
        if context_length + len(bbox_ids) >= context_length_limitation:
            left = context_length_limitation - context_length
            bbox_ids = bbox_ids[:left]
            bbox_token_tags = bbox_token_tags[:left]
            bbox_tokens = bbox_tokens[:left]
            block_token_bboxes=block_token_bboxes[:left]
            bbox_break = True

        one_bbox_len = len(bbox_ids)
        start = question_length+context_length
        end = start + one_bbox_len
        input_ids[0, start:end] = torch.from_numpy(np.array(bbox_ids))
        input_tags[0, start:end] = torch.from_numpy(np.array(bbox_token_tags))
        input_bbox_orders[0, start:end] = torch.from_numpy(np.array([bbox_order] * len(bbox_ids)))

        bbox_offsets[TAG_SELF[0]]['seq_offsets'] = [start, end]
        new_bbox_offsets = update_tag_seq_offsets(bbox_offsets, mode)

        bbox_length += one_bbox_len
        context_length += one_bbox_len

        new_block_metas.append((bbox_order, bbox_ids, bbox_tokens, bbox_token_tags, new_bbox_offsets, bbox_token2word_idx, bbox_words,word_bboxes,block_token_bboxes,bbox_lists))
        new_word_bboxes.append(word_bboxes)
        new_block_token_bboxes.extend(block_token_bboxes)
        if bbox_break:
            break
    bbox_mask[0, question_length:question_length + context_length] = 1
    input_ids[0, question_length+context_length:question_length + context_length + 1] = torch.from_numpy(np.array([sep_end]))
    input_bboxes[0, question_length:question_length + context_length + 1] = torch.from_numpy(np.array(new_block_token_bboxes[:context_length] + [[1000, 1000, 1000, 1000]]))
    input_segments[0, question_length] = 0
    input_segments[0, question_length:question_length + context_length + 1] = 1
    
    attention_mask = input_ids != 0

    return input_ids,input_bboxes, attention_mask, input_segments, question_mask, bbox_mask, input_bbox_orders,input_tags, new_question_metas, new_block_metas

def concat(question_metas,
            table_metas,
            paragraph_metas,
            sep_start,
            sep_end,
            max_pieces):

    question_ids, question_tokens, question_token_tags, question_offsets, question_token2word_idx, question_words = question_metas

    input_ids = torch.zeros([1, max_pieces])

    input_segments = torch.zeros_like(input_ids)
    input_row_ids = torch.zeros_like(input_ids)
    input_col_ids = torch.zeros_like(input_ids)
    input_para_orders = torch.zeros_like(input_ids)
    input_tags = torch.zeros_like(input_ids)

    question_mask = torch.zeros_like(input_ids)
    table_mask = torch.zeros_like(input_ids)
    paragraph_mask = torch.zeros_like(input_ids)

    # add question to the seq
    question_ids = [sep_start] + question_ids + [sep_end]
    question_token_tags = [0] + question_token_tags + [0]
    question_length = len(question_ids)
    input_ids[0, :question_length] = torch.from_numpy(np.array(question_ids))
    input_tags[0, :question_length] = torch.from_numpy(np.array(question_token_tags))
    question_mask[0, 1: question_length - 1] = 1 # remove the [start] and [end ]

    question_offsets[TAG_SELF[0]]['seq_offsets'] = [1, question_length - 1]
    new_question_offsets = update_tag_seq_offsets(question_offsets)

    new_question_metas = (question_ids, question_tokens, question_token_tags, new_question_offsets, question_token2word_idx, question_words)

    # add table to the seq
    context_length_limitation = max_pieces - 1 - question_length
    context_length = 0
    table_length = 0
    new_table_metas = []
    for cell_idx, (cell_row_ids, cell_col_ids, cell_ids, cell_tokens, cell_token_tags, cell_offsets, cell_token2word_idx, cell_words) in enumerate(table_metas):
        if context_length + len(cell_ids) > context_length_limitation:
            break
        cell_len = len(cell_ids)
        start = question_length + context_length
        end = start + cell_len
        input_ids[0, start:end] = torch.from_numpy(np.array(cell_ids))
        input_tags[0, start:end] = torch.from_numpy(np.array(cell_token_tags))
        input_row_ids[0, start:end] = torch.from_numpy(np.array(cell_row_ids))
        input_col_ids[0, start:end] = torch.from_numpy(np.array(cell_col_ids))

        cell_offsets[TAG_SELF[0]]['seq_offsets'] = [start, end]
        new_cell_offsets = update_tag_seq_offsets(cell_offsets)

        table_length += cell_len
        context_length += cell_len
        new_table_metas.append((cell_row_ids, cell_col_ids, cell_ids, cell_tokens, cell_token_tags, new_cell_offsets, cell_token2word_idx, cell_words))

    table_mask[0, question_length: question_length + table_length] = 1
    input_ids[0, question_length+table_length:question_length + table_length + 1] = torch.from_numpy(np.array([sep_end]))
    context_length += 1
    table_length += 1

    # add paragraphs to the seq
    paragraphs_length = 0
    new_paragraph_metas = []
    para_break = False

    #
    if context_length < context_length_limitation:
        for para_idx, (para_order, paragraph_ids, paragraph_tokens, paragraph_token_tags, paragraph_offsets, paragraph_token2word_idx, paragraph_words) in enumerate(paragraph_metas):
            if context_length + len(paragraph_ids) >= context_length_limitation:
                left = context_length_limitation - context_length
                paragraph_ids = paragraph_ids[:left]
                paragraph_token_tags = paragraph_token_tags[:left]
                paragraph_tokens = paragraph_tokens[:left]
                para_break = True
                if 'self_answer_offsets' in paragraph_offsets[TAG_SELF[0]]:
                    a=paragraph_offsets[TAG_SELF[0]]['self_answer_offsets']
                    print(a)

            one_para_len = len(paragraph_ids)
            start = question_length + context_length
            end = start + one_para_len
            input_ids[0, start:end] = torch.from_numpy(np.array(paragraph_ids))
            input_tags[0, start:end] = torch.from_numpy(np.array(paragraph_token_tags))
            input_para_orders[0, start:end] = torch.from_numpy(np.array([para_order] * len(paragraph_ids)))

            paragraph_offsets[TAG_SELF[0]]['seq_offsets'] = [start, end]
            new_paragraph_offsets = update_tag_seq_offsets(paragraph_offsets)

            paragraphs_length += one_para_len
            context_length += one_para_len

            new_paragraph_metas.append((para_order, paragraph_ids, paragraph_tokens, paragraph_token_tags, new_paragraph_offsets, paragraph_token2word_idx, paragraph_words))
            if para_break:
                break
        paragraph_mask[0, question_length + table_length:question_length + context_length] = 1
        input_ids[0, question_length + context_length:question_length + context_length + 1] = torch.from_numpy(np.array([sep_end]))

    attention_mask = input_ids != 0

    return input_ids, attention_mask, input_segments, question_mask, table_mask, paragraph_mask, \
           input_row_ids, input_col_ids, input_para_orders, input_tags, new_question_metas, new_table_metas, new_paragraph_metas


def sort_paragraphs(question:str, paragraphs: list, gold_paragraph_orders=None):

    paragraph_map = {p['order']: p['text'] for p in paragraphs}

    # apply tf-idf to calculate text-similarity
    sorted_order = get_order_by_tf_idf(question, paragraph_map)

    def order_key(p):
        order = p['order']
        if gold_paragraph_orders and order in gold_paragraph_orders:
            return 0
        else:
            return sorted_order.index(p['order'])
    sorted_paragraphs = sorted(paragraphs, key=lambda p: order_key(p))

    return sorted_paragraphs


def get_span_pos(tags):
    seq_len = tags.shape[1]
    begin_pos = 0
    end_pos = 0
    for i in range(seq_len):
        if tags[0, i] == 0:
            continue
        else:
            begin_pos = i
            break
    for i in range(seq_len - 1, -1, -1):
        if tags[0, i] == 0:
            continue
        else:
            end_pos = i
            break

    return begin_pos, end_pos


def overlap1d(r1, r2):
    return r1[1] >= r2[0] and r2[1] >= r1[0]


def block_tokenize_no_bbox(question, block_map, tokenizer, mapping, num_para_pos):
    split_tokens = []
    split_tags = []
    number_mask = []
    tokens = []
    tags = []
    word_piece_mask = []
    block_index = []
    block_num_pos = []
    
    answer_uuid_idx_mapping = defaultdict(list)
    
    for one in mapping:
        for uuid, idx in one.items():
            answer_uuid_idx_mapping[uuid].append(idx)
        
    # apply tf-idf to calculate text-similarity
    sorted_uuids = get_sorted_block_uuids(question, block_map)

    num_para_pos = list(set(num_para_pos))
    num_para_pos_c = num_para_pos.copy()
    num_para_pos.sort(key=lambda x: (sorted_uuids.index(x[0]), x[1]))
    num_para_order = [num_para_pos.index(_) for _ in num_para_pos_c]
    num_para_pos.reverse()
    num_para_index = []
    current_index = 0
    for uuid in sorted_uuids:
        text = block_map[uuid]['text']
        prev_is_whitespace = True
        answer_indexs = None
        if str(uuid) in answer_uuid_idx_mapping:
            answer_indexs = answer_uuid_idx_mapping[uuid]
        current_tags = [0 for i in range(len(text))]

        if answer_indexs is not None:
            for answer_index in answer_indexs:
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]

        start_index = 0
        wait_add = False
        for i, c in enumerate(text):
            if num_para_pos and num_para_pos[-1][0] == uuid:
                if i == num_para_pos[-1][1]:
                    num_para_index.append([current_index, current_index])
                if i == num_para_pos[-1][2] -1: # end is out
                    num_para_index[-1][-1] = current_index
                    num_para_pos.pop()
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                current_index += 1
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    current_index += 1
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)

    try:
        assert len(tokens) == len(tags)
    except AssertionError:
        print(len(tokens), len(tags))
        input()
    current_token_index = 1

    for i, token in enumerate(tokens):
        if i != 0:
            sub_tokens = tokenizer._tokenize(" " + token)
        else:
            sub_tokens = tokenizer._tokenize(token)

        for sub_token in sub_tokens:
            split_tags.append(tags[i])
            split_tokens.append(sub_token)
            block_index.append(current_token_index)
        current_token_index += 1
        word_piece_mask += [1]
        if len(sub_tokens) > 1:
            word_piece_mask += [0] * (len(sub_tokens) - 1)
    block_ids = tokenizer.convert_tokens_to_ids(split_tokens)
    
 
    for s_order in num_para_order:
        try:
            assert s_order < len(num_para_index)
        except AssertionError:
            print(s_order, len(num_para_index))
            input()
        block_num_pos.append(num_para_index[s_order])

    return tokens, block_ids, split_tags, word_piece_mask, number_mask, block_index, block_num_pos


def block_tokenize(question, block_map, tokenizer, mapping, num_para_pos):
    words = []
    tokens = []
    token_tags = []
    block_word_index = []
    block_num_pos_dict = {}
    token_bboxes = []
    
    answer_uuid_idx_mapping = defaultdict(list)
    for one in mapping:
        for uuid, idx in one.items():
            answer_uuid_idx_mapping[uuid].append(idx)
        
    # apply tf-idf to calculate text-similarity
    sorted_uuids = get_sorted_block_uuids(question, block_map)

    num_para_pos = list(set(num_para_pos))
    num_para_pos_dict = defaultdict(list)
    for idx, one in enumerate(num_para_pos):
        num_para_pos_dict[one[0]].append((idx, [one[1], one[2]])) # uuid -> val
    
    current_word_index = 1 # must start from 1 
    for uuid in sorted_uuids:
        text = block_map[uuid]['text']
        words_list = block_map[uuid]['words']['word_list']
        bbox_list = block_map[uuid]['words']['bbox_list']

        assert len(text.split()) == len(words_list)

        word_idxes = []
        cur = 0
        for i, w in enumerate(words_list):
            end = cur + len(w)
            if i == 0:
                word_idxes.append((0, end))
            else:
                word_idxes.append((cur, end))
            cur = end + 1 # space
        
        assert len(words_list) == len(word_idxes)

        for i, (word, bbox, w_s_e) in enumerate(zip(words_list, bbox_list, word_idxes)):
            words.append(word)
            word_tokens = tokenizer._tokenize(word)
            if len(word_tokens) == 0:
                continue
            # if has overlap 
            tag = 0
            if uuid in answer_uuid_idx_mapping:
                for one_idx in answer_uuid_idx_mapping[uuid]:
                    if overlap1d(one_idx, w_s_e):
                        tag = 1
                if uuid in num_para_pos_dict:
                    for num_order, one_num_idx in num_para_pos_dict[uuid]:
                        one_num_start, one_num_end = one_num_idx[0], one_num_idx[1]
                        key = f'{num_order}'
                        if one_num_start >= w_s_e[0] and one_num_start <= w_s_e[1]:
                            block_num_pos_dict[key] = [current_word_index] # the first token of the word 
                        if one_num_end >= w_s_e[0] and one_num_end <= w_s_e[1]:
                            block_num_pos_dict[key].append(current_word_index+1) 
                        
            for token in word_tokens:
                tokens.append(token)
                token_tags.append(tag)
                token_bboxes.append(bbox)
                block_word_index.append(current_word_index)
                # current_token_index += 1
            current_word_index += 1
        
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    block_word_num_pos = sorted([(num_order, num_pos) for num_order, num_pos in block_num_pos_dict.items()]) 
    block_word_num_pos = [one[1] for one in block_word_num_pos]
    return words, tokens, token_bboxes, token_ids, token_tags, block_word_index, block_word_num_pos
