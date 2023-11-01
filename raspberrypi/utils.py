import time
import numpy

def repeat_n_times_and_analysis(n):
    def repeat_helper(func, *args, **kwargs):
        times = []
        for _ in range(n):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            times.append(end - start)
        total = sum(times)
        average = total/n
        print(f"Total time   : {total} sec")
        print(f"Max time     : {max(times)} sec")
        print(f"Min time     : {min(times)} sec")
        print(f"Average time : {average} sec")
        print(f"STD DEV      : {numpy.std(times)}")
    return repeat_helper


def get_energy():
    with open('watt.csv', 'r') as rf:
        lines = rf.readlines()
    usable_lines = lines[1:]
    prev = None
    ans = 0
    for line in usable_lines:
        t, p = line.split(',')[:2]
        if prev is not None:
            ans += (int(t) - prev) / 1000 * float(p)
        prev = int(t)
    return ans
