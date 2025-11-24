
# utils/derivatives.py

def derivative(poly, var_idx, vars):
    res = {}
    for mon, coeff in poly.items():
        md = dict(mon)
        var = vars[var_idx]
        pwr = md.get(var, 0)
        if pwr:
            new = list(mon)
            new = [(x, e) for x, e in new if x != var]
            if pwr > 1:
                new.append((var, pwr - 1))
            new.sort()
            res[tuple(new)] = res.get(tuple(new), 0) + coeff * pwr
    return res

def multivariate_derivative(poly, multi, vars):
    res = poly
    for idx, times in enumerate(multi):
        for _ in range(times):
            res = derivative(res, idx, vars)
            if not res:
                return {}
    return res
