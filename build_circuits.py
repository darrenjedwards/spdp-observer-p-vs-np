
# build_circuits.py
import math
import random

def build_majority_poly(n):
    vars = [f"x{i}" for i in range(n)]
    poly = {}
    for _ in range(n):
        group = random.sample(vars, n // 2 + 1)
        poly[tuple((v, 1) for v in group)] = 1
    return poly, vars

def build_addressing_poly(n):
    addr_bits = math.ceil(math.log2(n))
    vars = [f"x{i}" for i in range(n)]
    addr_vars = [f"a{i}" for i in range(addr_bits)]
    poly = {}
    for i in range(n):
        addr = [(addr_vars[j], (i >> j) & 1) for j in range(addr_bits)]
        poly[tuple([(vars[i], 1)] + addr)] = 1
    return poly, vars + addr_vars

def build_parity_poly(n):
    vars = [f"x{i}" for i in range(n)]
    poly = {tuple((v, 1) for v in vars): 1}
    return poly, vars

def build_randdeg3_poly(n):
    vars = [f"x{i}" for i in range(n)]
    poly = {}
    for _ in range(n):
        clause = random.sample(vars, 3)
        poly[tuple((v, 1) for v in clause)] = 1
    return poly, vars

def build_crvw_extractor(n):
    assert n % 3 == 0, "n must be divisible by 3"
    vars = [f"x{i}" for i in range(n)]
    poly = {}
    for i in range(0, n, 3):
        group = vars[i:i+3]
        poly[tuple((v, 1) for v in group)] = 1
    return poly, vars

def build_goldreich_prf(n, d=3):
    vars = [f"x{i}" for i in range(n)]
    poly = {}
    for _ in range(n):
        group = random.sample(vars, d)
        poly[tuple((v, 1) for v in group)] = 1
    return poly, vars

def build_diagonal_poly(n, d=3):
    vars = [f"x{i}" for i in range(n)]
    poly = {}
    for v in vars:
        poly[((v, d),)] = 1
    return poly, vars
