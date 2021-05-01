# distutils: language=c++

import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map

def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

cpdef target_mean_v3(data, y_name, x_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow, dtype=np.float64))
    cdef np.ndarray[double] y = np.asfortranarray(data[y_name], dtype=np.float64)
    cdef np.ndarray[int] x = np.asfortranarray(data[x_name], dtype=np.int32)

    target_mean_v3_impl(result, y, x, nrow)
    return result


cdef void target_mean_v3_impl(double[:] result, double[:] y, int[:] x, const long nrow):
    cdef unordered_map[int, double] value_dict
    cdef unordered_map[int, int] count_dict
    cdef unordered_map[int, double] mean_dict

    for i in range(nrow):
        value_dict[x[i]] += y[i]
        count_dict[x[i]] += 1

    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i]) / (count_dict[x[i]]-1)