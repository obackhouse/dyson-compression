'''
Miscalleneous functions.
'''


def prange(start, stop=None, step=1):
    ''' Loop over boundaries for blocks in a given range.
    '''

    if stop is None:
        start, stop = 0, start

    for p0 in range(start, stop, step):
        p1 = min(p0+step, stop)
        yield p0, p1
