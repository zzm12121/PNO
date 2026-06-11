class GenerateEquation:
    def __init__(self, expr, coeff_dict):
        self.expr = expr
        self.coeff_dict = coeff_dict


class GeneratorDetails:
    def __init__(self, fixed_flag,cfg):
        if fixed_flag:
            self.variables = cfg.variables
            self.constants = cfg.constants
        else:
            self.max_len = cfg.max_len
            self.operators = cfg.operators
            self.max_ops = cfg.max_ops
            self.rewrite_functions = cfg.rewrite_functions
            self.variables = cfg.variables
            self.eos_index = cfg.eos_index
            self.pad_index = cfg.pad_index
            self.priori_leaf = cfg.priori_leaf




class ConstDetails:
    def __init__(self, fixed_flag,cfg):
        if fixed_flag:
            self.num_sample = cfg.num_of_sample
            self.repeat = cfg.repeat
            self.num_constant = cfg.num

            self.constant=cfg.constant_range
            # self.constant_min = [cfg.constant_range[n][0] for n in range(self.num_constant)]
            # self.constant_max = [cfg.constant_range[n][1] for n in range(self.num_constant)]
        else:
            self.num_sample = cfg.num_of_sample
            self.repeat = cfg.repeat
            self.constant_min = cfg.constant_range[0]
            self.constant_max = cfg.constant_range[1]




