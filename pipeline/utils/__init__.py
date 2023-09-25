import time

def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Elapsed time: {time.time()-start} seconds")
        return result
    return wrapper
