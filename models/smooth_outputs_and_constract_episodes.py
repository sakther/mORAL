import numpy as np
import pandas as pd


def smooth_output(T, D, Y):
    Tn = list(T)
    dur = list(D)
    Yin = [v for v in list(Y)]
    Yout = [0 for _ in list(Y)]

    i = 0

    while i < len(Yin):
        if Yin[i] == 1:
            j = i + 1
            cur_dur = dur[i]
            while Tn[i] <= Tn[j] <= Tn[i] + dur[i] + 90:
                if Yin[j] == 1:
                    cur_dur += dur[j]
                j += 1
            if cur_dur >= 45:
                for k in range(i, j):
                    Yout[k] = Yin[k]
                i += 1
            else:
                i += 1
        else:
            i += 1
    return Yout


