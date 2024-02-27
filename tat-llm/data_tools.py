import re
from copy import deepcopy


def normalize_derivation(s):
    p = re.compile('[£€ ,\$a-zA-Z]')
    return re.sub(p, '', s).replace('–', '-')


def proderivation(de):
    p = re.search('(?<=-\()[\d\.\+]+\d%?(?=\)|/)', de)
    if p:
        new = re.sub('(?<=\+)(?=[1-9]|0\.)', '-', de[p.start():p.end()])
        if p.start() > 2:
            return de[0:p.start() - 2] + '+(-' + new + de[p.end():]
        return '(-' + new + de[p.end():]
    return de


def change2neg(s):
    while re.search('\((\d+)\)', s):
        p = re.search('\((\d+)\)', s)
        s = s[:p.start()] + '-' + s[p.start() + 1:p.end() - 1] + s[p.end():]
    return s


def dele(out):
    i = 0
    while i + 2 < len(out):
        if out[i] == '(' and re.search('\d', out[i + 1]) and out[i + 2] == ')':
            out = out[:i] + [out[i + 1]] + out[i + 3:]
        i += 1
    return out


def tag(out, nums):
    out_seq = []
    consts = []
    for n in out:
        if nums.count(n) > 0:
            out_seq.append("N" + str(nums.index(n)))
            # if n not in out_dict.keys():
            #     out_dict[n]=nums.index(n)
        else:
            out_seq.append(n)
            #verify the const
            if n+'0' in nums:
                continue
            if n not in ['+','-','*', '/', '**', '>']:
                consts.append(n)
    return out_seq, consts


def get_nums_from_derivation(deri):
    nums = []
    patten = re.compile('(((?<![\d%\)])-)?\d+\.\d+%?)|(((?<![\d%\)])-)?\d+%?)')
    t = re.finditer(patten, deri)
    for i in t:
        if i.group() not in nums:
            nums.append(i.group())
    return nums


def segment(st, nums, nums_neg):  # seg the equation and tag the num
    res = []
    for n in nums_neg:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += segment(st[:p_start], nums, nums_neg)
                if res and '0' <= res[-1][-1] <= '9':
                    res.append('+')
            if nums.count(n) == 1:
                res.append("N" + str(nums.index(n)))
            else:
                res.append(n)
            if p_end < len(st):
                behind = segment(st[p_end:], nums, nums_neg)
                if behind and '0' <= behind[0][-1] <= '9':
                    res.append('+')
                res += behind
            return res
    st = normalize_derivation(st)
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += segment(st[:p_start], nums, nums_neg)
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append("N" + str(nums.index(st_num)))
        else:
            res.append(st_num)
        if p_end < len(st):
            res += segment(st[p_end:], nums, nums_neg)
        return res
    for ss in st:
        if ss != ' ':
            res.append(ss)

    return dele(res)


def seg(st, nums, nums_neg):  # seg the equation and tag the num
    res = []
    for n in nums_neg:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg(st[:p_start], nums, nums_neg)
                if res and '0' <= res[-1][-1] <= '9':
                    res.append('+')
            if nums.count(n) == 1:
                res.append(n)
            else:
                res.append(n)
            if p_end < len(st):
                behind = seg(st[p_end:], nums, nums_neg)
                if behind and '0' <= behind[0][-1] <= '9':
                    res.append('+')
                res += behind
            return res
    st = normalize_derivation(st)
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg(st[:p_start], nums, nums_neg)
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append(st_num)
        else:
            res.append(st_num)
        if p_end < len(st):
            res += seg(st[p_end:], nums, nums_neg)
        return res
    for ss in st:
        if ss != ' ':
            res.append(ss)

    return dele(res)


def get_num_pos(nums, table_res, table_pos, para_res, para_pos):
    num_table_pos = []
    num_para_pos = []
    num_table = []
    num_para = []
    for num in nums:
        num_p = None
        for i, tbn in enumerate(table_res):
            if num_p not in num_table_pos:
                if tbn == num or abs(eval(re.sub('%', '', tbn)) - eval(re.sub('%', '', num))) < 1e-4:
                    num_p = table_pos[i]
                    num_table.append(num)
                    num_table_pos.append(num_p)
                    break
                elif len(num) > 1:
                    if len(tbn) > len(num):
                        if tbn[:len(num)] == num or tbn[len(tbn)-len(num):] == num:
                            num_p = table_pos[i]
                            num_table.append(num)
                            num_table_pos.append(num_p)
                            break
                    else:
                        if num[:len(tbn)] == tbn or num[len(num)-len(tbn):] == tbn:
                            num_p = table_pos[i]
                            num_table.append(num)
                            num_table_pos.append(num_p)
                            break
        if num_p:
            continue
        for i, prn in enumerate(para_res):
            if prn == num or abs(eval(re.sub('%', '', prn)) - eval(re.sub('%', '', num))) < 1e-4:
                num_p = para_pos[i]
                num_para.append(num)
                num_para_pos.append(num_p)
                break
            elif len(num) > 1:
                if len(prn) > len(num):
                    if table_pos and (prn[:len(num)] == num or prn[len(prn) - len(num):] == num):
                        num_p = table_pos[i]
                        num_table.append(num)
                        num_table_pos.append(num_p)
                        break
                else:
                    if table_pos and (num[:len(prn)] == prn or num[len(num) - len(prn):] == prn):
                        num_p = table_pos[i]
                        num_table.append(num)
                        num_table_pos.append(num_p)
                        break
        if num_p:
            continue
    if not num_table and not num_para:
        for num in nums:
            num_p = None
            for i, tbn in enumerate(table_res):
                if tbn in num or num in tbn:
                    num_p = table_pos[i]
                    num_table.append(num)
                    num_table_pos.append(num_p)
                    break
            if num_p:
                continue
            for i, prn in enumerate(para_res):
                if prn in num or num in prn:
                    num_p = para_pos[i]
                    num_para.append(num)
                    num_para_pos.append(num_p)
                    break
            if num_p:
                continue
    if not num_table and not num_para:
        num_table, num_table_pos = table_res + para_res, table_pos + para_pos

    return num_table, num_table_pos, num_para, num_para_pos


def get_mapping_pos(table, paragraphs, mapping):
    table_nums = []
    table_pos = []
    para_nums = []
    para_pos = []
    pattern = re.compile("\d+\.\d+%?|\d+%?")
    neg_pat = re.compile("\((\d+\.\d+%?|\d+%?)\)")
    for key in mapping.keys():
        if key == 'table':
            table_idexes = mapping['table']
            for index in table_idexes:
                seg = table[index[0]][index[1]].strip().split(" ")
                for s in seg:
                    s = normalize_derivation(s)
                    neg_p = re.search(neg_pat, s)
                    p = re.search(pattern, s)
                    if neg_p:
                        table_nums.append('-' + neg_p.group()[1:-1])
                        table_pos.append((index[0], index[1]))
                    elif p:
                        table_nums.append(p.group())
                        table_pos.append((index[0], index[1]))
        if key in ['paragraph', 'paragraphs']:
            for order in mapping[key].keys():
                indexes = mapping[key][order]
                for index in indexes:
                    seg = paragraphs[int(order) - 1]['text'][index[0]:index[1]].strip().split(" ")
                    begin = index[0]
                    for s in seg:
                        s = normalize_derivation(s)
                        neg_p = re.search(neg_pat, s)
                        p = re.search(pattern, s)
                        if neg_p:
                            para_nums.append('-' + neg_p.group()[1:-1])
                            para_pos.append((int(order), begin + neg_p.start(), begin + neg_p.end()))
                        if p:
                            para_nums.append(p.group())
                            para_pos.append((int(order), begin + p.start(), begin + p.end()))
                        begin += len(s)
    return table_nums, table_pos, para_nums, para_pos


def istree(l):
    infix = l.copy()
    infix.reverse()
    head = infix.pop()
    flag = True

    def dfs(root):
        nonlocal flag
        if root not in ['-', '+', '*', '/']:
            return
        if infix:
            left = infix.pop()
            dfs(left)
        else:
            flag = False
            return
        if infix:
            right = infix.pop()
            dfs(right)
        else:
            flag = False
            return

    dfs(head)
    return flag


def seg_and_tag(st, nums, nums_neg):  # seg the equation and tag the num
    res = []
    for n in nums_neg:
        if n in st:
            p_start = st.find(n)
            p_end = p_start + len(n)
            if p_start > 0:
                res += seg_and_tag(st[:p_start], nums, nums_neg)
                if res and '0' <= res[-1][-1] <= '9':
                    res.append('+')
            if nums.count(n) == 1:
                res.append("N" + str(nums.index(n)))
            else:
                res.append(n)
            if p_end < len(st):
                behind = seg_and_tag(st[p_end:], nums, nums_neg)
                if behind and '0' <= behind[0][-1] <= '9':
                    res.append('+')
                res += behind
            return res
    st = normalize_derivation(st)
    pos_st = re.search("\d+\.\d+%?|\d+%?", st)
    if pos_st:
        p_start = pos_st.start()
        p_end = pos_st.end()
        if p_start > 0:
            res += seg_and_tag(st[:p_start], nums, nums_neg)
        st_num = st[p_start:p_end]
        if nums.count(st_num) == 1:
            res.append("N" + str(nums.index(st_num)))
        else:
            res.append(st_num)
        if p_end < len(st):
            res += seg_and_tag(st[p_end:], nums, nums_neg)
        return res
    for ss in st:
        if ss != ' ':
            res.append(ss)

    return res


def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def get_from_mapping(table, paragraphs, mapping):
    def toneg(num):
        return re.sub('^\(', '-', num.strip(')'))

    num_table, num_table_pos, num_para, num_para_pos = [], [], [], []
    for key in mapping.keys():
        if key == 'table':
            table_idexes = mapping['table']
            for index in table_idexes:
                num = toneg(normalize_derivation(table[index[0]][index[1]]))
                if num not in num_table and num not in num_para:
                    num_table.append(num)
                    num_table_pos.append((index[0], index[1]))
        if key in ['paragraph', 'paragraphs']:
            for order in mapping[key].keys():
                indexes = mapping[key][order]
                for index in indexes:
                    num = toneg(normalize_derivation(paragraphs[int(order) - 1]['text'][index[0]::index[1]]))
                    if num not in num_para and num not in num_table:
                        num_para.append(num)
                        num_para_pos.append((int(order), index[0], index[1]))

    return num_table, num_table_pos, num_para, num_para_pos
