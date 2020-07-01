import itertools
import random
### warnings errors ###

def warning(message):
    print("WARNING: {}".format(message))

def error(message):
    print("ERROR: {}".format(message))
    exit(0)

# ordinary operations
def flatten2d(list2d):
    return list(itertools.chain(*list2d))

def flatten3d(list3d):
    return flatten2d(flatten2d(list3d))

def flip(weight):
    '''
    draw the true / false value according to a given weight
    '''
    tmp_rd = random.random()
    return tmp_rd <= weight