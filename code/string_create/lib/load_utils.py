import json
from string_create.lib import class_utils
from string_create.lib import generate_utils


def create_env(fixed_flag,cfg):
    param = class_utils.GeneratorDetails(fixed_flag,cfg)
    env = generate_utils.Generator(fixed_flag,param)
    return env


def const(fixed_flag,cfg):
    param = class_utils.ConstDetails(fixed_flag,cfg)
    return param




