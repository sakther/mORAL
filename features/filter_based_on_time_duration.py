

MIN_SEGMENT_DURATION = 15 # seconds
MAX_SEGMENT_DURATION = 4*60 # seconds

MIN_SEGMENT_DURATION_flossing = 2 # seconds
# MAX_SEGMENT_DURATION = 3*60 # seconds

MIN_SEGMENT_DURATION_rinsing = 0.5 # seconds
MAX_SEGMENT_DURATION_rinsing = 3 # seconds

INTER_CAND_DIFF = 5 # sec

def filter_based_on_duration(cands, AGMO, mindur = MIN_SEGMENT_DURATION):
    filtered_cands = []
    for cand in cands:
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO[start_index][0]
        etime = AGMO[end_index][0]
        dur = etime - stime
        if dur >= mindur and dur <= MAX_SEGMENT_DURATION:
            filtered_cands.append(cand)
    return filtered_cands

def filter_for_brushing(cands, AGMO):
    cands = filter_based_on_duration(cands, AGMO, mindur=5)
    cands = merge_candidates(cands, AGMO)
    cands = filter_based_on_duration(cands, AGMO, mindur=MIN_SEGMENT_DURATION)
    # cands = filter_based_on_duration(cand_f, AGMO)

    return cands



def filter_based_on_max_duration_rinsing(cands, AGMO):
    filtered_cands = []
    for cand in cands:
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO[start_index][0]
        etime = AGMO[end_index][0]
        dur = etime - stime
        if dur >= MIN_SEGMENT_DURATION_rinsing and dur <= MAX_SEGMENT_DURATION_rinsing:
            filtered_cands.append(cand)
    return filtered_cands

def filter_based_on_min_duration_flossing(cands, AGMO):
    filtered_cands = []
    for cand in cands:
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO[start_index][0]
        etime = AGMO[end_index][0]
        dur = etime - stime
        if dur >= MIN_SEGMENT_DURATION_flossing:
            filtered_cands.append(cand)
    return filtered_cands

def merge_candidates(cands, AGMO):
    filtered_cands = []
    if len(cands) ==0:
        return filtered_cands
    cur_cand = [cands[0][0], cands[0][1]]
    i = 1
    while i< len(cands):
        cand = cands[i]
        start_index = cand[0]
        end_index = cand[1]
        stime = AGMO[start_index][0]
        etime = AGMO[end_index][0]
        if stime - AGMO[cur_cand[1]][0] <= INTER_CAND_DIFF:
            cur_cand[1] = end_index
        else:
            filtered_cands.append(cur_cand)
            cur_cand = [start_index, end_index]
        i+=1
    filtered_cands.append(cur_cand)
    return filtered_cands

def get_overlap_portion(st1, et1, st2, et2):
    overlap_portion = min(et1, et2) - max(st1, st2)
    return overlap_portion


def combine_left_right(cand_l, cand_r):

    cands = []
    for c_l in cand_l:
        for c_r in cand_r:
            overlap = get_overlap_portion(c_l[0], c_l[1], c_r[0], c_r[1])
            if overlap > 0:
                cands.append([max(c_l[0], c_r[0]), min(c_l[1], c_r[1])])
    return cands


