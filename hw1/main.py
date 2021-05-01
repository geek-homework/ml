# coding = 'utf-8'
import tm
import time
import numpy as np
import pandas as pd


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def main():
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    start = time.time()
    result_1 = target_mean_v1(data, 'y', 'x')
    print("1", time.time() - start)
    start = time.time()
    result_2 = tm.target_mean_v2(data, 'y', 'x')
    print("2", time.time() - start)
    start = time.time()
    result_3 = tm.target_mean_v3(data, 'y', 'x')
    print("3", time.time() - start)

    print(np.linalg.norm(result_1 - result_2))
    print(np.linalg.norm(result_2 - result_3))


if __name__ == '__main__':
    main()