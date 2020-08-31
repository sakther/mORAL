
# left_ori = {0:[1,1,1], 1:[1,1,1], 2:[-1,-1,1], 3:[-1,1,1], 4:[1,-1,1]}
# right_ori = {0:[1,-1,1], 1:[1,-1,1], 2:[-1,1,1], 3:[-1,-1,1], 4:[1,1,1]}
#
# left_ori_newdevice = {0:[-1,1,1], 1:[-1,1,1], 2:[1,-1,1], 3:[1,1,1], 4:[-1,-1,1]}
# right_ori_newdevice = {0:[1,1,1], 1:[1,1,1], 2:[-1,-1,1], 3:[-1,1,1], 4:[1,-1,1]}
# right_ori_newdevice = {0:[1,-1,1], 1:[1,-1,1], 2:[-1,1,1], 3:[-1,-1,1], 4:[1,1,1]}

left_ori = {0:[1,1,1], 1:[-1,-1,1], 2:[-1,1,-1], 3:[1,-1,-1]}
# right_ori = {0:[1,-1,1], 1:[-1,1,1], 2:[-1,-1,1], 3:[1,1,1]}
right_ori = left_ori

left_ori_newdevice = {0:[-1,1,1], 1:[1,-1,1], 2:[1,1,-1], 3:[-1,-1,-1]}
# right_ori_newdevice = {0:[1,1,1], 1:[-1,-1,1], 2:[-1,1,1], 3:[1,-1,1]}
right_ori_newdevice = left_ori_newdevice


def change_orientation(pid, cur_ori):
    # if int(pid) <= 10:
    #     return cur_ori
    #
    # if cur_ori == 0 or cur_ori == 1:
    #     return 2
    # elif cur_ori == 2:
    #     return 1
    # elif cur_ori == 4:
    #     return 3
    # elif cur_ori == 3:
    #     return 4
    return cur_ori

def used_new_device(pid):
    if int(pid) <= 10:
        return False
    return True
