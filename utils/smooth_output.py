import numpy as np

def do_smooth_output(T, Y):
    Yn = [v for v in list(Y)]

    for i in range(3, len(Y)-3):
        val = np.mean([Y[j] for j in range(i-3, i+3)])
        if val>=0.5:
            Yn[i] = 1
        else:
            Yn[i]=0

    return Yn

def do_smooth_output_new(T, Y):
    Yn = [v for v in list(Y)]

    for i in range(3, len(Y)-3):
        val = np.mean([Y[j] for j in range(i-1, i+1)])
        if val>=0.5:
            Yn[i] = 1
        else:
            Yn[i]=0

    return Yn

