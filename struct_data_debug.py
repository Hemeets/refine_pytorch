'''
Author: QDX
Date: 2022-10-19 11:07:06
Description: 
'''
import torch
import numpy
import pandas


def debug_tensor(t: torch.tensor, name='', show_val=False):
    print('* ' * 10 + name + ' *' * 10)
    print("[shape]: {}, [dtype]: {}".format(
        t.shape, t.dtype
    ))
    print("[storage_offset]: {}, [stride]: {}, [contiguous]: {}".format(
         t.storage_offset(), t.stride(), t.is_contiguous()
    ))
    if show_val:
        print("[val]: {}".format(t))
    print('- ' * 30)
    return None


def debug_array(t: numpy.array, name='', show_val=False):
    print('* ' * 10 + name + ' *' * 10)
    print("[shape]: {}, [dtype]: {}".format(
        t.shape, t.dtype
    ))
    if show_val:
        print("[val]: {}".format(t))
    print('- ' * 30)
    return None


def debug_pd_series(d: pandas.Series, name='', show_val=False):
    print('* ' * 10 + name + ' *' * 10)
    print("[shape]: {}, [dtype]: {}".format(
        d.shape, d.dtype
    ))
    print("[index]: {}".format(d.index))
    if show_val:
        print("[val]: {}".format(d.values))
    print('- ' * 30)
    return None


def debug_pd_df(d: pandas.DataFrame, name='', show_val=False):
    print('* ' * 10 + name + ' *' * 10)
    print("[shape]: {}".format(d.shape))
    print("[dtypes]:\n{}".format(d.dtypes))
    print("[index]: {}".format(d.index))
    # print("[columns]: {}".format(d.columns))
    if show_val:
        print("[val]:\n{}".format(d.values))
    print('- ' * 30)
    return None



if __name__ == "__main__":
    # pass
    import pandas as pd
    # data = pd.Series([0.25, 0.5, 0.75, 1.0])
    # debug_pd_series(data, show_val=True)
    data = pd.DataFrame([0.25, 0.5, 0.75, 1.0])
    debug_pd_df(data)
