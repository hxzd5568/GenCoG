import typing as t
from typing import Any, Callable

from colorama import Fore, Back

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, Filter, \
    InSet, Subset
from .basic import Expr, Const, Var, Range, Symbol, Env, Arith, Cmp, Not, And, Or, ForAll, Cond, \
    GetAttr, Dummy
from .tensor import Num, TensorDesc, Shape, Rank, GetDType
from .visitor import ExprVisitor
from ..util import CodeBuffer, NameGenerator, cls_name, colored_text


class ExprPrinter(ExprVisitor[Env[str], Any]):
    def __init__(self, buf: CodeBuffer, highlights: t.List[Expr]):
        super().__init__()
        self._buf = buf
        self._high_ids = set(id(e) for e in highlights)
        self._name_gen = NameGenerator('_s', [])
        self._color_idx = 0

    def visit(self, expr: Expr, env: Env[str]):
        high = id(expr) in self._high_ids
        if high:
            self._buf.write(Back.RED + Fore.BLACK)
        super().visit(expr, env)
        if high:
            self._buf.write(Back.RESET + Fore.RESET)

    def visit_const(self, const: Const, env: Env[str]):
        self._buf.write(str(const.val_))

    def visit_var(self, var: Var, env: Env[str]):
        self._write_cls(var)
        items = []
        if var.type_ is not None:
            items.append(('ty', var.type_, lambda ty: self._buf.write(str(ty))))
        if var.ran_ is not None:
            items.append(('ran', var.ran_, lambda ran: self.visit(ran, env)))
        self._write_named(items)

    def visit_range(self, ran: Range, env: Env[str]):
        self._write_cls(ran)
        items = []
        if ran.begin_ is not None:
            items.append(('begin', ran.begin_, lambda beg: self.visit(beg, env)))
        if ran.end_ is not None:
            items.append(('end', ran.end_, lambda end: self.visit(end, env)))
        self._write_named(items)

    def visit_symbol(self, sym: Symbol, env: Env[str]):
        self._buf.write(f'{cls_name(sym)}(\'{env[sym]}\')')

    colors = [
        Fore.YELLOW,
        Fore.GREEN,
        Fore.BLUE,
    ]

    def _next_color(self):
        color = self.colors[self._color_idx]
        self._color_idx = (self._color_idx + 1) % len(self.colors)
        return color

    def visit_arith(self, arith: Arith, env: Env[str]):
        color = self._next_color()
        self._write_pos(
            [
                (arith.lhs_, lambda lhs: self.visit(lhs, env)),
                (arith.op_, lambda op: self._buf.write(op.value)),
                (arith.rhs_, lambda rhs: self.visit(rhs, env))
            ],
            sep=' ',
            prefix=colored_text('(', color),
            suffix=colored_text(')', color)
        )

    def visit_cmp(self, cmp: Cmp, env: Env[str]):
        color = self._next_color()
        self._write_pos(
            [
                (cmp.lhs_, lambda lhs: self.visit(lhs, env)),
                (cmp.op_, lambda op: self._buf.write(op.value)),
                (cmp.rhs_, lambda rhs: self.visit(rhs, env))
            ],
            sep=' ',
            prefix=colored_text('(', color),
            suffix=colored_text(')', color)
        )

    def visit_not(self, n: Not, env: Env[str]):
        self._write_cls(n)
        self._write_pos([n.prop_, lambda prop: self.visit(prop, env)])

    def visit_and(self, a: And, env: Env[str]):
        self._write_cls(a)
        self._write_pos_multi([(cl, lambda c: self.visit(c, env)) for cl in a.clauses_])

    def visit_or(self, o: Or, env: Env[str]):
        self._write_cls(o)
        self._write_pos_multi([(cl, lambda c: self.visit(c, env)) for cl in o.clauses_])

    def visit_forall(self, forall: ForAll, env: Env[str]):
        nested_env = self._gen_nested_env(env, forall.idx_)
        self._write_cls(forall)
        self._write_named_multi([
            ('ran', forall.ran_, lambda ran: self.visit(ran, env)),
            ('idx', forall.idx_, lambda idx: self.visit_symbol(idx, nested_env)),
            ('body', forall.body_, lambda body: self.visit(body, nested_env))
        ])

    def visit_cond(self, cond: Cond, env: Env[str]):
        self._write_cls(cond)
        self._write_named_multi([
            ('pred', cond.pred_, lambda pred: self.visit(pred, env)),
            ('tr_br', cond.tr_br_, lambda br: self.visit(br, env)),
            ('fls_br', cond.fls_br_, lambda br: self.visit(br, env))
        ])

    def visit_attr(self, attr: GetAttr, env: Env[str]):
        self._buf.write(f'a(\'{attr.name_}\')')

    def visit_dummy(self, dum: Dummy, env: Env[str]):
        self._buf.write(f'{cls_name(dum)}()')

    def visit_num(self, num: Num, env: Env[str]):
        self._buf.write(f'{num.t_kind_.value}.num')

    def visit_shape(self, shape: Shape, env: Env[str]):
        self._write_tensor_attr(shape.tensor_, 'shape', env)

    def visit_rank(self, rank: Rank, env: Env[str]):
        self._write_tensor_attr(rank.tensor_, 'rank', env)

    def visit_dtype(self, dtype: GetDType, env: Env[str]):
        self._write_tensor_attr(dtype.tensor_, 'dtype', env)

    def _write_tensor_attr(self, desc: TensorDesc, name: str, env: Env[str]):
        self._buf.write(f'{desc.kind_.value}[')
        self.visit(desc.idx_, env)
        self._buf.write(f'].{name}')

    def visit_tuple(self, tup: Tuple, env: Env[str]):
        self._write_cls(tup)
        self._write_pos([(f, lambda f: self.visit(f, env)) for f in tup.fields_])

    def visit_list(self, lst: List, env: Env[str]):
        nested_env = self._gen_nested_env(env, lst.idx_)
        self._write_cls(lst)
        self._write_named_multi([
            ('len', lst.len_, lambda l: self.visit(l, env)),
            ('idx', lst.idx_, lambda idx: self.visit_symbol(idx, nested_env)),
            ('body', lst.body_, lambda body: self.visit(body, nested_env))
        ])

    def visit_getitem(self, getitem: GetItem, env: Env[str]):
        self.visit(getitem.arr_, env)
        self._write_pos([(getitem.idx_, lambda idx: self.visit(idx, env))], prefix='[', suffix=']')

    def visit_len(self, ln: Len, env: Env[str]):
        self._write_cls(ln)
        self._write_pos([(ln.arr_, lambda arr: self.visit(arr, env))])

    def visit_concat(self, concat: Concat, env: Env[str]):
        self._write_cls(concat)
        self._write_pos_multi([(arr, lambda arr: self.visit(arr, env)) for arr in concat.arrays_])

    def visit_slice(self, slc: Slice, env: Env[str]):
        self.visit(slc.arr_, env)
        self._write_pos([(slc.ran_, lambda ran: self.visit(ran, env))], prefix='[', suffix=']')

    def visit_map(self, m: Map, env: Env[str]):
        nested_env = self._gen_nested_env(env, m.sym_)
        self._write_cls(m)
        self._write_named_multi([
            ('arr', m.arr_, lambda arr: self.visit(arr, env)),
            ('sym', m.sym_, lambda sym: self.visit(sym, nested_env)),
            ('body', m.body_, lambda body: self.visit(body, nested_env))
        ])

    def visit_reduce_array(self, red: ReduceArray, env: Env[str]):
        self._write_cls(red)
        self._write_named_multi([
            ('arr', red.arr_, lambda arr: self.visit(arr, env)),
            ('op', red.op_, lambda op: self._buf.write(op.value)),
            ('init', red.init_, lambda init: self.visit(init, env))
        ])

    def visit_reduce_index(self, red: ReduceIndex, env: Env[str]):
        nested_env = self._gen_nested_env(env, red.idx_)
        self._write_cls(red)
        self._write_named_multi([
            ('ran', red.ran_, lambda ran: self.visit(red.ran_, env)),
            ('op', red.op_, lambda op: self._buf.write(op.value)),
            ('idx', red.idx_, lambda idx: self.visit(idx, nested_env)),
            ('body', red.body_, lambda body: self.visit(body, nested_env)),
            ('init', red.init_, lambda init: self.visit(init, env))
        ])

    def visit_filter(self, flt: Filter, env: Env[str]):
        nested_env = self._gen_nested_env(env, flt.sym_)
        self._write_cls(flt)
        self._write_named_multi([
            ('arr', flt.arr_, lambda arr: self.visit(arr, env)),
            ('sym', flt.sym_, lambda sym: self.visit(sym, nested_env)),
            ('pred', flt.pred_, lambda pred: self.visit(pred, nested_env))
        ])

    def visit_inset(self, inset: InSet, env: Env[str]):
        self._write_cls(inset)
        self._write_named_multi([
            ('elem', inset.elem_, lambda elem: self.visit(elem, env)),
            ('set', inset.set_, lambda s: self.visit(s, env))
        ])

    def visit_subset(self, subset: Subset, env: Env[str]):
        self._write_cls(subset)
        self._write_named_multi([
            ('sub', subset.sub_, lambda sub: self.visit(sub, env)),
            ('sup', subset.sup_, lambda sup: self.visit(sup, env))
        ])

    def _gen_nested_env(self, env: Env[str], sym: Symbol):
        name = self._name_gen.generate()
        return env + (sym, name)

    def _write_cls(self, e: Expr):
        self._buf.write(cls_name(e))

    def _write_pos(self, items: t.List[t.Tuple[Any, Callable[[Any], None]]],
                   sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self._buf.write_pos(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_pos_multi(self, items: t.List[t.Tuple[Any, Callable[[Any], None]]],
                         sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self._buf.write_pos_multi(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_named(self, items: t.List[t.Tuple[str, Any, Callable[[Any], None]]],
                     sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self._buf.write_named(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_named_multi(self, items: t.List[t.Tuple[str, Any, Callable[[Any], None]]],
                           sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self._buf.write_named_multi(items, sep=sep, prefix=prefix, suffix=suffix)


def print_expr(expr: Expr, buf: CodeBuffer, highlights: t.List[Expr]):
    ExprPrinter(buf, highlights).visit(expr, Env())
