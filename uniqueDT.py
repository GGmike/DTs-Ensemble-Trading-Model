from itertools import combinations

def uniqueDT(data, key=None):
    columns = data.columns if key is None else [key]
    unique_rows = set()