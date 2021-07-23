import time

def elapsed_time(func):
    def inner():
        start = time.time()
        func()
        end = time.time()
        print('function: ', func.__name__)
        print('elapsed time: ' ,end-start, 'seconds')
    return inner


def test():
    print('This is a test function')

test = elapsed_time(test)
test()