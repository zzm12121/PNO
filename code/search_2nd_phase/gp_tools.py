#    -----------------------------------------------------------------
#    This code is copied from open-source software DEAP (https://github.com/DEAP/deap/blob/master/deap/gp.py)
#    and make our customized modifications as needed.
#                                       ——Jiaxu Cui, 2023-12
#    -----------------------------------------------------------------

"""The :mod:`gp` module provides the methods and classes to perform
Genetic Programming with DEAP. It essentially contains the classes to
build a Genetic Program Tree, and the functions to evaluate it.
"""
import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from inspect import isclass

import os
import sys
import contextlib

from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt

import numpy as np
import numpy.random

from deap import *
import sympy as sp

from torch_scatter import scatter_sum
import torch
import scipy.integrate as spi


######################################
# GP Data structure                  #
######################################

# Define the name of type for any types.
__type__ = object


class PrimitiveTree(list):
    """Tree specifically formatted for optimization of genetic programming
    operations. The tree is represented with a list, where the nodes are
    appended, or are assumed to have been appended when initializing an object
    of this class with a list of primitives and terminals e.g. generated with
    the method **gp.generate**, in a depth-first order.
    The nodes appended to the tree are required to have an attribute *arity*,
    which defines the arity of the primitive. An arity of 0 is expected from
    terminals nodes.
    """

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5,"
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")
        list.__setitem__(self, key, val)

    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        return string

    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))

                expr.append(Terminal(token, False, type_))

        return cls(expr)

    ##############################################################
    @classmethod
    def from_string_sympy(cls, string_sympy, pset):  # implemented by Jiaxu Cui at 2023-12-24
        preorder_list = []
        #print(string_sympy, sp.simplify(string_sympy))
        build_tree(sp.simplify(string_sympy), preorder_list)
        ind_str = "".join(preorder_list)
        return cls.from_string(ind_str, pset)

    ##############################################################
    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)


class Primitive(object):
    """Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.

        >>> pr = Primitive("mul", (int, int), int)
        >>> pr.format(1, 2)
        'mul(1, 2)'
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        args = ", ".join(map("{{{0}}}".format, range(self.arity)))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Terminal(object):
    """Class that encapsulates terminal primitive in expression. Terminals can
    be values or 0-arity functions.
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class MetaEphemeral(type):
    """Meta-Class that creates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated.
    """
    cache = {}

    def __new__(meta, name, func, ret=__type__, id_=None):
        if id_ in MetaEphemeral.cache:
            return MetaEphemeral.cache[id_]

        if isinstance(func, types.LambdaType) and func.__name__ == '<lambda>':
            warnings.warn("Ephemeral {name} function cannot be "
                          "pickled because its generating function "
                          "is a lambda function. Use functools.partial "
                          "instead.".format(name=name), RuntimeWarning)

        def __init__(self):
            self.value = func()

        attr = {'__init__': __init__,
                'name': name,
                'func': func,
                'ret': ret,
                'conv_fct': repr}

        cls = super(MetaEphemeral, meta).__new__(meta, name, (Terminal,), attr)
        MetaEphemeral.cache[id(cls)] = cls
        return cls

    def __init__(cls, name, func, ret=__type__, id_=None):
        super(MetaEphemeral, cls).__init__(name, (Terminal,), {})

    def __reduce__(cls):
        return (MetaEphemeral, (cls.name, cls.func, cls.ret, id(cls)))


copyreg.pickle(MetaEphemeral, MetaEphemeral.__reduce__)


class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix="ARG"):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        # setting "__builtins__" to None avoid the context
        # being polluted by builtins function when evaluating
        # GP expression.
        self.context = {"__builtins__": None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        self.name = name
        self.ret = ret_type
        self.ins = in_types
        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)

        assert name not in self.context or \
               self.context[name] is primitive, \
            "Primitives are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second '%s' primitive." % (name,)

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        assert name not in self.context, \
            "Terminals are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        if name not in self.mapping:
            class_ = MetaEphemeral(name, ephemeral, ret_type)
        else:
            class_ = self.mapping[name]
            if class_.func is not ephemeral:
                raise Exception("Ephemerals with different functions should "
                                "be named differently, even between psets.")
            if class_.ret is not ret_type:
                raise Exception("Ephemerals with the same name and function "
                                "should have the same type, even between psets.")

        self._add(class_)
        self.terms_count += 1

    def addADF(self, adfset):
        """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
        prim = Primitive(adfset.name, adfset.ins, adfset.ret)
        self._add(prim)
        self.prims_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix="ARG"):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, "arity should be >= 1"
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)

    def addTerminal(self, terminal, name=None):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)


######################################
# GP Tree compilation functions      #
######################################
def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)


def compileADF(expr, psets):
    """Compile the expression represented by a list of trees. The first
    element of the list is the main tree, and the following elements are
    automatically defined functions (ADF) that can be called by the first
    tree.


    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param psets: List of primitive sets. Each set corresponds to an ADF
                  while the last set is associated with the expression
                  and should contain reference to the preceding ADFs.
    :returns: a function if the main primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    adfdict = {}
    func = None
    for pset, subexpr in reversed(list(zip(psets, expr))):
        pset.context.update(adfdict)
        func = compile(subexpr, pset)
        adfdict.update({pset.name: func})
    return func


######################################
# GP Program generation functions    #
######################################
def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def genRamped(pset, min_, max_, type_=None):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


######################################
# GP Crossovers                      #
######################################

def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # Determine whether to keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        # Set does not support indexing
        type_ = random.sample(common_types, 1)[0]
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


######################################
# GP Mutations                       #
######################################
def mutUniform(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        term = random.choice(pset.terminals[node.ret])
        if type(term) is MetaEphemeral:
            term = term()
        individual[index] = term
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


def mutEphemeral(individual, mode):
    """This operator works on the constants of the tree *individual*. In
    *mode* ``"one"``, it will change the value of one of the individual
    ephemeral constants by calling its generator function. In *mode*
    ``"all"``, it will change the value of **all** the ephemeral constants.

    :param individual: The normal or typed tree to be mutated.
    :param mode: A string to indicate to change ``"one"`` or ``"all"``
                 ephemeral constants.
    :returns: A tuple of one tree.
    """
    if mode not in ["one", "all"]:
        raise ValueError("Mode must be one of \"one\" or \"all\"")

    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(type(node), MetaEphemeral)]

    if len(ephemerals_idx) > 0:
        if mode == "one":
            ephemerals_idx = (random.choice(ephemerals_idx),)

        for i in ephemerals_idx:
            individual[i] = type(individual[i])()

    return individual,


def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,


def mutShrink(individual):
    """This operator shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).

    :param individual: The tree to be shrunk.
    :returns: A tuple of one tree.
    """
    # We don't want to "shrink" the root
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    iprims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            iprims.append((i, node))

    if len(iprims) != 0:
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree

    return individual,


######################################
#   Our customized modifications     #
######################################


######################################
# GP Tree compilation functions      #
######################################

def compile_torch(expr, pset, constants_symbols):
    """Compile the expression *expr* for torch.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments) + ',' + ','.join(c for c in constants_symbols)
        code = "lambda {args}: {code}".format(args=args, code=code)
        code = code.replace('\"', '')
        code = code.replace('\'', '')
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)




def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# def solve_ivp_with_timeout(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
#     try:
#         with stdout_redirected():  # Ignoring warning errors encountered in solving initial value problems
#             soluation_Y, t_range = solve_ivp(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01)
#     except func_timeout.exceptions.FunctionTimedOut:
#         print('solve_ivp func_timeout ... ')
#         t_range = np.arange(t_start, t_end + t_inc, t_inc)
#         soluation_Y = np.repeat(X0.reshape(1, -1, X0.shape[-1]), t_range.shape[0],  axis=0)
#     return soluation_Y, t_range
#
# @func_set_timeout(5.)
def solve_ivp(eval_func_f, eval_func_g, X0, sparse_A, t_start=0, t_end=1, t_inc=0.01):
    N, x_dim = X0.shape
    row, col = sparse_A
    
    """
    def diff_func(x, t):
        # dx_i(t)/dt = func(x)
        x_j = x.reshape(-1, x_dim)[row]
        x_i = x.reshape(-1, x_dim)[col]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=1e30)
        diff_g = np.nan_to_num(eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=1e30)

        if len(diff_f.shape) < 2:
            diff_f = x.reshape(-1, x_dim)[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j[:, :1]

        # print(diff_f.shape, diff_g.shape)
        # assert diff_g.shape[1] == 1
        # assert diff_g.shape[0] == x_i_j.shape[0]

        # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))

        dX = np.array(diff_f, dtype=np.float) + \
             scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float)), torch.from_numpy(col).long().view(-1,1), dim=0,
                         dim_size=x.reshape(-1, x_dim).shape[0]).numpy()

        return dX.reshape(-1)

    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    # lock.acquire()
    # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-12, atol=1e-12)
    # New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-9, atol=1e-9)
    New_X = spi.odeint(diff_func, X0.reshape(-1), t_range, rtol=1e-3, atol=1e-6)
    
    """
    def diff_func(t, x):
        # dx_i(t)/dt = func(x)
        x_j = x.reshape(-1, x_dim)[row]
        x_i = x.reshape(-1, x_dim)[col]
        x_i_j = np.concatenate([x_i, x_j], axis=-1)

        # we do not know the scatter_sum in numpy package, so we use scatter_sum in torch instead.
        diff_f = np.nan_to_num(eval_func_f(*[x.reshape(-1, x_dim)[:, i].reshape(-1, 1) for i in range(x_dim)]), nan=1e30)
        diff_g = np.nan_to_num(eval_func_g(*[x_i_j.reshape(-1, x_dim + x_dim)[:, i].reshape(-1, 1) for i in range(x_dim + x_dim)]), nan=1e30)

        if len(diff_f.shape) < 2:
            diff_f = x.reshape(-1, x_dim)[:, :1]
        if len(diff_g.shape) < 2:
            diff_g = x_i_j[:, :1]

        # print(diff_f.shape, diff_g.shape)
        # assert diff_g.shape[1] == 1
        # assert diff_g.shape[0] == x_i_j.shape[0]

        # print(np.any(np.isnan(diff_f)), np.any(np.isnan(diff_g)))

        dX = np.array(diff_f, dtype=np.float) + \
             scatter_sum(torch.from_numpy(np.array(diff_g, dtype=np.float)), torch.from_numpy(col).long().view(-1,1), dim=0,
                         dim_size=x.reshape(-1, x_dim).shape[0]).numpy()

        return dX.reshape(-1)
        
    t_range = np.arange(t_start, t_end + t_inc, t_inc)
    
    sol = spi.solve_ivp(diff_func, (min(t_range), max(t_range)), X0.reshape(-1), t_eval=t_range, dense_output=True)#, method='RK45', rtol=1e-3, atol=1e-6)
    
    New_X = sol.sol(t_range).T
    
    
    """
    if sol.status != 0:
        New_X = np.zeros((len(t_range), N, x_dim))
    else:
        New_X = sol.y.T
    """
        
    #print(New_X.status)
    #exit(1)
    #New_X = sol.y.T
    #New_X = sol.sol(t_range).T
    
    
    # lock.release()

    return New_X.reshape(len(t_range), N, x_dim), t_range.reshape(-1, 1)



def print_expression(e, level=0):
    spaces = " " * level
    if isinstance(e, (sp.Symbol, sp.Number)):
        print(spaces + str(e))
        return
    if len(e.args) > 0:
        print(spaces + e.func.__name__)
        for arg in e.args:
            print_expression(arg, level + 1)
    else:
        print(spaces + e.func.__name__)


def build_tree(expr, preorder_list):
    if expr.func.__name__ == 'Add' or expr.func.__name__ == 'Mul':
        for _ in range(len(expr.args) - 1):
            preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list)
        for i in range(len(expr.args) - 1):
            preorder_list.append(', ')
            build_tree(expr.args[i + 1], preorder_list)
            preorder_list.append(')')
    elif expr.func.__name__ == 'Sub':
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list)
        preorder_list.append(', ')
        build_tree(expr.args[1], preorder_list)
        preorder_list.append(')')
    elif expr.func.__name__ == 'Div':
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list)
        preorder_list.append(', ')
        build_tree(expr.args[1], preorder_list)
        preorder_list.append(')')
    elif expr.func.__name__ == 'Pow':
        # preorder_list.append('%s(' % expr.func.__name__)
        # build_tree(expr.args[0], preorder_list)
        # preorder_list.append(', ')
        # build_tree(expr.args[1], preorder_list)
        # preorder_list.append(')')

        # we only deal with pow(x, y) whose y is integer
        # print(expr.args[1])
        if float(expr.args[1]) % 1 == 0:
            if expr.args[1] < 0:
                preorder_list.append('Div(')
                preorder_list.append('1')
                preorder_list.append(', ')
                for _ in range(int(abs(expr.args[1])) - 1):
                    preorder_list.append('Mul(')
                build_tree(expr.args[0], preorder_list)
                for i in range(int(abs(expr.args[1])) - 1):
                    preorder_list.append(', ')
                    build_tree(expr.args[0], preorder_list)
                    preorder_list.append(')')
                preorder_list.append(')')
            elif expr.args[1] == 0:
                preorder_list.append('1')
            elif expr.args[1] > 0:
                for _ in range(int(expr.args[1]) - 1):
                    preorder_list.append('Mul(')
                build_tree(expr.args[0], preorder_list)
                for i in range(int(expr.args[1]) - 1):
                    preorder_list.append(', ')
                    build_tree(expr.args[0], preorder_list)
                    preorder_list.append(')')
            else:
                print('Unknown %s in Pow' % expr.args[1])
                exit(1)
        else:
            preorder_list.append('Pow(')
            build_tree(expr.args[0], preorder_list)
            preorder_list.append(', ')
            preorder_list.append('%s)'%expr.args[1])
        
    elif expr.func.__name__ == 'exp' or expr.func.__name__ == 'sin' or expr.func.__name__ == 'cos':
        preorder_list.append('%s(' % expr.func.__name__)
        build_tree(expr.args[0], preorder_list)
        preorder_list.append(')')
    elif isinstance(expr, (sp.Symbol, sp.Number)):
        preorder_list.append(str(expr))
    elif expr is sp.zoo:
        preorder_list.append(str(1.0))
    elif expr.func.__name__ == 'Pi':
        preorder_list.append(str(3.1415926))
    else:
        print('Unknown %s' % expr.func.__name__)
        exit(1)


def mut_Terminal(individual, pset, sampling_const):
    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(node, Terminal)]

    if len(ephemerals_idx) > 0:
        ephemerals_idx = random.choice(ephemerals_idx)
        # if not isinstance(individual[ephemerals_idx].value, str):
        #     constant
        individual[ephemerals_idx] = random.choice(
            [Terminal(sampling_const(), False, object), pset.terminals[object][0]])
        # individual[ephemerals_idx] = Terminal(sampling_const(), False, object)

    return individual


def mut_Operator(individual, pset):
    if len(individual) < 2:
        return individual

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        # do nothing
        return individual
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)
    return individual


def mut_InsertNode(individual, pset):
    return mutInsert(individual, pset)[0]


def mut_SubtreeShrink(individual, pset):
    return mutShrink(individual)[0]


"""
An example of converter:
    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'cos': lambda x: sp.cos(x),
        'inv': lambda x: x ** (-1),
        'sqrt': lambda x: sp.sqrt(x),
    }
"""


def TreeToDeadStr(ind, converter):
    # return sp.sympify(str(ind), locals=converter)
    return sp.parse_expr(str(ind), local_dict=converter)


# def DeadStrToTree(read_str, converter):
#     return sp.sympify(str(ind), locals=converter)

def mut_Simplify(individual, pset, converter):
    eq_str_reduced = str(sp.simplify(TreeToDeadStr(individual, converter)))
    if eq_str_reduced is 'nan' or 'zoo' in eq_str_reduced:
        return individual
    else:
        return PrimitiveTree.from_string_sympy(eq_str_reduced, pset)


def mut_NewTree(individual, pset, min_, max_):
    return PrimitiveTree(genGrow(pset, min_, max_, type_=None))


def Mutations(individual, pset, sampling_const, converter, min_, max_, eval_func=None, x=None, y=None):
    individual_new = copy.deepcopy(individual)

    case = random.choice([1, 2, 3, 4, 5, 6])
    # print('case=', case)
    # case = 5
    if case == 1:
        # mutate constant or param
        individual_new = mut_Terminal(individual_new, pset, sampling_const)  # test ok
    elif case == 2:
        # mutate operator
        individual_new = mut_Operator(individual_new, pset)  # test ok
    elif case == 3:
        # mutate : insert node
        individual_new = mut_InsertNode(individual_new, pset)  # test ok
    elif case == 4:
        # mutate : shrink subtree
        individual_new = mut_SubtreeShrink(individual_new, pset)  # test ok
    elif case == 5:
        # mutate : simplify tree
        # individual_new = mut_Simplify(individual_new, pset, converter)  # too cost in time, so we ignore this mutation
        individual_new = individual_new
    elif case == 6:
        # mutate : new tree entirely
        individual_new = mut_NewTree(individual_new, pset, min_, max_)  # test ok
    else:
        print('Unknown case [%s] in Mutations!!!' % case)
        exit(1)

    if eval_func is None or x is None or y is None:
        return individual_new

    old_complex = len(individual)
    old_fitness = eval_func(individual, pset, x, y)

    new_complex = len(individual_new)
    new_fitness = eval_func(individual_new, pset, x, y)

    alpha = 0.1
    annealing_temperature = 1.0

    q_anneal = np.exp(-(new_fitness - old_fitness) / (alpha * annealing_temperature))
    q_parsimony = float(old_complex / new_complex)

    if random.uniform(0, 1) < q_anneal * q_parsimony:
        return individual_new
    else:
        return individual


def Mutations_f_g(individual_f_g, pset, sampling_const, converter, min_, max_, eval_func=None, x=None, y=None):
    pset_f_, pset_g_ = pset
    individual_f_new = Mutations(individual_f_g[0], pset_f_, sampling_const, converter, min_, max_, eval_func=eval_func,
                                 x=x, y=y)
    individual_g_new = Mutations(individual_f_g[1], pset_g_, sampling_const, converter, min_, max_, eval_func=eval_func,
                                 x=x, y=y)
    return (individual_f_new, individual_g_new)


