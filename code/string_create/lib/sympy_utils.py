import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _simplify(f)



def remove_root_constant_terms(expr, variables, mode):
    """
    Remove root constant terms from a non-constant SymPy expression.
    """
    variables = variables if type(variables) is list else [variables]
    assert mode in ["add", "mul", "pow"]
    if not any(x in variables for x in expr.free_symbols):
        return expr
    if mode == "add" and expr.is_Add or mode == "mul" and expr.is_Mul:
        args = [
            arg
            for arg in expr.args
            if any(x in variables for x in arg.free_symbols) or (arg in [-1])
        ]
        if len(args) == 1:
            expr = args[0]
        elif len(args) < len(expr.args):
            expr = expr.func(*args)
    elif mode == "pow" and expr.is_Pow:
        assert len(expr.args) == 2
        if not any(x in variables for x in expr.args[0].free_symbols):
            return expr.args[1]
        elif not any(x in variables for x in expr.args[1].free_symbols):
            return expr.args[0]
        else:
            return expr
    return expr


def remove_mul_const(f, variables):
    """
    Remove the multiplicative factor of an expression, and return it.
    """
    if not f.is_Mul:
        return f, 1
    variables = variables if type(variables) is list else [variables]
    var_args = []
    cst_args = []
    for arg in f.args:
        if any(var in arg.free_symbols for var in variables):
            var_args.append(arg)
        else:
            cst_args.append(arg)
    return sp.Mul(*var_args), sp.Mul(*cst_args)