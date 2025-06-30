import numpy as np

def default_chars(rows, cols):
    base = [chr(i) for i in range(65, 91)] + [str(i) for i in range(0, 10)]
    base += ['<', '>', '_', '.', ',', '?']
    return base[:rows * cols]

def generate_flash_sequence(rows, cols, n_flashes, flash_mode):
    seq = []
    for _ in range(n_flashes):
        if flash_mode == 'row/col':
            s = [('row', i) for i in range(rows)] + [('col', j) for j in range(cols)]
        elif flash_mode == 'checkerboard':
            s = [('checker', (i, j)) for i in range(rows) for j in range(cols) if (i+j)%2==0]
        elif flash_mode == 'region':
            region_size = 3
            s = []
            for i in range(0, rows, region_size):
                for j in range(0, cols, region_size):
                    s.append(('region', (i, j)))
        else:
            s = [('single', idx) for idx in range(rows * cols)]
        np.random.shuffle(s)
        seq.extend(s)
    return seq
