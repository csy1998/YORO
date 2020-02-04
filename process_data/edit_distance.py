# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fei_wang@shannonai.com
@Time       : 2019/8/18 11:11
@Description:
"""


def min_edit_distance_path(s1, s2):
    # s1: src, s2: tgt
    s1 = ["#"] + s1
    s2 = ["#"] + s2

    # init distance matrix
    dis = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dis[i][0] = i
    for j in range(len(s2) + 1):
        dis[0][j] = j
    pre_id = [[(-1, -1) for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]

    # update distance matrix
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            if dis[i - 1][j - 1] + cost <= dis[i - 1][j] and dis[i - 1][j - 1] + cost <= dis[i][j - 1]:
                dis[i][j] = dis[i - 1][j - 1] + cost
                pre_id[i][j] = (i - 1, j - 1)
            elif dis[i - 1][j] <= dis[i][j - 1]:
                dis[i][j] = dis[i - 1][j] + 1
                pre_id[i][j] = (i - 1, j)
            else:
                dis[i][j] = dis[i][j - 1] + 1
                pre_id[i][j] = (i, j - 1)

    # restore path
    path = []
    tmp_i, tmp_j = len(s1), len(s2)
    while tmp_i != -1 and tmp_j != -1:
        path.append((tmp_i, tmp_j))
        tmp_i, tmp_j = pre_id[tmp_i][tmp_j]
    final_path = []
    path.pop()
    for point in path[::-1]:
        final_path.append((point[0] - 1, point[1] - 1))
    return final_path


def edit_path_to_correction(src, tgt, path, enable_replace=False):
    # path [n, m] n=len(src)+1, m=len(tgt)+1
    # from (0,0) to (n,m)
    # right: add
    # down: delete
    # right-down: replace (or not)
    correction = []  # list of (error type, position, new character)
    src = ["#"] + src
    tgt = ["#"] + tgt
    pre_i, pre_j = 0, 0
    for i, j in path[1:]:
        if i > pre_i and j > pre_j:
            if src[i] != tgt[j]:
                if enable_replace:
                    correction.append(("replace", i - 1, tgt[j]))
                else:
                    correction.append(("delete", i - 1, ""))
                    correction.append(("add", i - 1, tgt[j]))
        elif i > pre_i:
            correction.append(("delete", i - 1, ""))
        else:
            correction.append(("add", i, tgt[j]))
        pre_i, pre_j = i, j
    return correction
