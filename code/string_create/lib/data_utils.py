import random
from torch.distributions.uniform import Uniform
from typing import Tuple


def sample_symbolic_constants(fixed_flag, eq, cfg=None) -> Tuple:
    if fixed_flag:
        dummy_consts = {const: 1 for const in eq.coeff_dict.keys()}
        consts = dummy_consts.copy()
        consts_data = {}
        for i in range(len(cfg.constant)):
            consts_data[cfg.constant[i][0]] = cfg.constant[i][1:]
        symbols_used = set(eq.coeff_dict.keys())

        for si in symbols_used:
            if consts_data[si][0] == 0:
                if len(consts_data[si]) == 1:
                    consts[si] = round(float(consts_data[si][1]), 3)
                else:
                    n = random.randint(1, len(consts_data[si]) - 1)
                    consts[si] = round(float(consts_data[si][n]), 3)
            else:
                consts[si] = round(float(Uniform(consts_data[si][1], consts_data[si][-1]).sample()), 3)

    else:
        dummy_consts = {const: 1 if const[:2] == "cm" else 0 for const in eq.coeff_dict.keys()}
        consts = dummy_consts.copy()
        symbols_used = set(eq.coeff_dict.keys())
        for si in symbols_used:
            if si[:2] == "cm":
                consts[si] = round(float(Uniform(cfg.constant_min, cfg.constant_max).sample()), 3)
            else:
                raise KeyError
    return consts, dummy_consts
