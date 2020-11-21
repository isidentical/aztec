from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from functools import partial, singledispatchmethod
from typing import DefaultDict, Dict, Iterable, List, Optional, TypeVar, Union


class CompilerError(Exception):
    ...


class BlockType(Enum):
    Class = auto()
    Module = auto()
    Function = auto()


class ScopeType(Enum):
    CELL = auto()
    FREE = auto()
    LOCAL = auto()
    GLOBAL_IMPLICIT = auto()
    GLOBAL_EXPLICIT = auto()


class ScopeFlags(IntFlag):
    USED = auto()
    FREE = auto()
    LOCAL = auto()
    PARAM = auto()
    GLOBAL = auto()
    ANNOTED = auto()
    NONLOCAL = auto()
    IMPORTED = auto()
    COMP_ITER = auto()
    FREE_CLASS = auto()

    BOUND = LOCAL | PARAM | IMPORTED


class FutureFeatures(IntFlag):
    NESTED_SCOPES = auto()
    GENERATORS = auto()
    DIVISION = auto()
    ABSOLUTE_IMPORT = auto()
    WITH_STATEMENT = auto()
    PRINT_FUNCTION = auto()
    UNICODE_LITERALS = auto()
    BARRY_AS_FLUFL = auto()
    GENERATOR_STOP = auto()
    ANNOTATIONS = auto()


T = TypeVar("T")

FlagType = Union[IntFlag, int]
Symbols = DefaultDict[str, FlagType]
Analysis = Dict[str, ScopeType]

_symbol_field = field(default_factory=partial(defaultdict, int))


def mangle(namespace_name: Optional[str], name: str) -> str:
    def prefix():
        return name[:2]

    def suffix():
        return name[-2:]

    # - If it is not a member of a class
    # - If it doesn't start with __
    # - If it does ends with __ (like __dunder__)
    if namespace_name is None or prefix() != "__" or suffix() == "__":
        return name

    namespace_name = namespace_name.lstrip("_")
    # - If it is all underscores, don't mangle
    if len(namespace_name) == 0:
        return name

    return "_" + namespace_name + name


@dataclass
class SymbolTable:
    filename: str

    module: SymbolTableEntry
    current: SymbolTableEntry

    stack: List[SymbolTableEntry] = field(default_factory=list)
    blocks: Dict[ast.AST, SymbolTableEntry] = field(default_factory=dict)
    global_namespace: Symbols = _symbol_field

    future_flags: FlagType = 0
    _class_level_namespaces: List[str] = field(default_factory=list)

    @property
    def namespace(self):
        if len(self._class_level_namespaces) > 0:
            return self._class_level_namespaces[-1]
        else:
            return None

    @classmethod
    def from_scratch(cls, filename: str, module: ast.mod):
        top_entry = SymbolTableEntry("top", module, BlockType.Module)

        symbol_table = cls(filename, top_entry, top_entry)
        symbol_table.global_namespace = top_entry.symbols
        symbol_table.stack.append(top_entry)
        symbol_table.blocks[module] = top_entry
        symbol_table.build(module)
        symbol_table.stack.pop()

        assert len(symbol_table.stack) == 0
        return symbol_table

    @contextmanager
    def new_block(
        self, name: str, node: ast.AST, block_type: BlockType
    ) -> Generator[SymbolTableEntry, None, None]:
        try:
            previous = self.current
            self.current = SymbolTableEntry(name, node, block_type)
            self.stack.append(self.current)
            self.blocks[node] = self.current
            if block_type is BlockType.Class:
                self._class_level_namespaces.append(name)
            previous.children.append(self.current)
            if previous.block_type is BlockType.Function or previous.get_flag(
                "nested"
            ):
                self.current.set_flag("nested")
            yield self.current
        finally:
            if block_type is BlockType.Class:
                self._class_level_namespaces.pop()
            self.stack.pop()
            self.current = self.stack[-1]

    def build(self, module: ast.mod):
        if isinstance(module, (ast.Module, ast.Interactive)):
            self.visit_all(self.visit_stmt, module.body)
        elif isinstance(module, ast.Expression):
            self.visit_expr(module.body)
        self.analyze_entry(self.module, AnalysisState())

    def analyze_entry(
        self, entry: SymbolTableEntry, parent_state: AnalysisState
    ):
        state = AnalysisState()
        # Class namespaces gets inherited before
        # the name analysis
        if entry.block_type is BlockType.Class:
            parent_state.inherit(state)

        for symbol, flags in entry.symbols.items():
            self.analyze_name(entry, state, parent_state, symbol, flags)

        if entry.block_type is BlockType.Class:
            state.bound_namespace.add("__class__")
        elif entry.block_type is BlockType.Function:
            state.bound_namespace |= parent_state.local_namespace

        # Function and module namespaces gets inherited
        # after the name analysis
        if entry.block_type is not BlockType.Class:
            parent_state.inherit(state)

        for child in entry.children:
            child_state = self.analyze_entry(child, state)
            state.free_namespace |= child_state.free_namespace
            if entry.get_flag("free"):
                entry.set_flag("child_free")

        if entry.block_type is BlockType.Function:
            self.analyze_cell_vars(entry, state)
        elif entry.block_type is BlockType.Class:
            entry.set_flag("needs_class_closure")
            state.free_namespace.discard("__class__")

        self.analyze_free_vars(entry, state)
        entry.analysis.update(state.scopes)

        return state

    def analyze_name(
        self,
        entry: SymbolTableEntry,
        state: AnalysisState,
        parent_state: AnalysisState,
        symbol: str,
        flags: FlagType,
    ) -> None:
        is_module_level = entry.block_type is BlockType.Module
        if flags & ScopeFlags.GLOBAL:
            if flags & ScopeFlags.NONLOCAL:
                raise CompilerError

            state.scopes[symbol] = ScopeType.GLOBAL_EXPLICIT
            parent_state.global_namespace.add(symbol)
            parent_state.bound_namespace.discard(symbol)
        elif flags & ScopeFlags.NONLOCAL:
            if is_module_level:
                raise CompilerError
            if symbol not in parent_state.bound_namespace:
                raise CompilerError
            state.scopes[symbol] = ScopeType.FREE
            parent_state.free_namespace.add(symbol)
            entry.set_flag("free")
        elif flags & ScopeFlags.BOUND:
            state.scopes[symbol] = ScopeType.LOCAL
            parent_state.local_namespace.add(symbol)
            parent_state.global_namespace.discard(symbol)
        elif symbol in parent_state.bound_namespace and not is_module_level:
            state.scopes[symbol] = ScopeType.FREE
            parent_state.free_namespace.add(symbol)
        else:
            state.scopes[symbol] = ScopeType.GLOBAL_IMPLICIT
            if symbol not in parent_state.global_namespace and entry.get_flag(
                "nested"
            ):
                entry.set_flag("free")

    def analyze_cell_vars(
        self, entry: SymbolTableEntry, state: AnalysisState
    ) -> None:
        for symbol, scope_type in state.scopes.copy().items():
            if (
                scope_type is not ScopeType.LOCAL
                or symbol not in state.free_namespace
            ):
                continue
            state.scopes[symbol] = ScopeType.CELL
            state.free_namespace.discard(symbol)

    def analyze_free_vars(
        self, entry: SymbolTableEntry, state: AnalysisState
    ) -> None:
        for symbol in state.free_namespace:
            if symbol in state.scopes:
                flags = self.lookup(symbol, entry)
                if flags & (ScopeFlags.BOUND | ScopeFlags.GLOBAL):
                    self.add_definition(symbol, ScopeFlags.FREE_CLASS, entry)
                continue

            if entry.block_type is not BlockType.Module:
                state.scopes[symbol] = ScopeType.FREE

    def visit_all(
        self, visitor: Callable[[T], None], nodes: Iterable[Optional[T]]
    ) -> None:
        for node in nodes:
            if node is not None:
                visitor(node)

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            self.visit_stmt(node)
        elif isinstance(node, ast.expr):
            self.visit_expr(node)
        elif isinstance(node, ast.alias):
            self.visit_alias(node)
        elif isinstance(node, ast.excepthandler):
            self.visit_excepthandler(node)
        else:
            self.visit_all(self.visit, ast.iter_child_nodes(node))

    # TO-DO(low): implement the annotated assign special condition
    @singledispatchmethod
    def visit_stmt(self, node: ast.stmt) -> None:
        self.visit_all(self.visit, ast.iter_child_nodes(node))

    @singledispatchmethod
    def visit_expr(self, node: ast.expr) -> None:
        self.visit_all(self.visit, ast.iter_child_nodes(node))

    def visit_arg(self, node: ast.arg) -> None:
        self.add_definition(node.arg, ScopeFlags.PARAM)

    def visit_keyword(self, node: ast.keyword) -> None:
        self.visit_expr(node.value)

    def visit_excepthandler(self, node: ast.excepthandler) -> None:
        if node.type is not None:
            self.visit_expr(node.type)
        if node.name is not None:
            self.add_definition(node.name, ScopeFlags.LOCAL)
        self.visit_all(self.visit_stmt, node.body)

    def visit_alias(self, node: ast.alias) -> None:
        if node.name == "*":
            if self.current.block_type is not BlockType.Module:
                raise CompilerError
            else:
                return None

        name = node.asname if node.asname else node.name
        name, _, __ = name.partition(".")
        self.add_definition(name, ScopeFlags.IMPORTED)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        with self.current.on_flag("comprehension_target"):
            self.visit_expr(node.target)
        with self.current.on_flag("comprehension_iterator"):
            self.visit_expr(node.iter)
        self.visit_all(self.visit_expr, node.ifs)
        if node.is_async:
            self.set_flag("coroutine", True)

    def visit_comprehensions(self, node: ComprehensionT) -> None:
        outermost = node.generators[0]
        # Outermost iterator is evaluated in current scope
        with self.current.on_flag("comprehension_iterator"):
            self.visit_expr(outermost.iter)

        # name e.g: ListComp => listcomp
        with self.new_block(
            type(node).__name__.lower(), node, BlockType.Function
        ) as block:
            block.set_flag("coroutine", outermost.is_async)
            block.set_flag("comprehension")
            self.add_implicit_parameter(0)
            with self.current.on_flag("comprehension_target"):
                self.visit_expr(outermost.target)
                self.visit_all(self.visit_expr, outermost.ifs)
                self.visit_all(self.visit_comprehension, node.generators[1:])
                if block.get_flag("generator"):
                    raise CompilerError

            if isinstance(node, ast.GeneratorExp):
                block.set_flag("generator")

    # TO-DO(low): annotations
    def visit_arguments(self, node: ast.arguments) -> None:
        self.visit_all(
            self.visit_arg,
            node.posonlyargs
            + node.args
            + node.kwonlyargs
            + [node.vararg, node.kwarg],
        )

    def visit_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_async: bool = False,
    ) -> None:
        self.add_definition(node.name, ScopeFlags.LOCAL)
        self.visit_all(self.visit_expr, node.decorator_list)
        self.visit_all(
            self.visit_expr, node.args.defaults + node.args.kw_defaults
        )
        with self.new_block(node.name, node, BlockType.Function) as block:
            if is_async:
                block.set_flag("coroutine")

            self.visit_arguments(node.args)
            self.visit_all(self.visit_stmt, node.body)

        if node.returns:
            self.visit_expr(node.returns)

    @visit_stmt.register
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.add_definition(node.name, ScopeFlags.LOCAL)
        self.visit_all(self.visit_expr, node.bases)
        self.visit_all(self.visit_keyword, node.keywords)
        self.visit_all(self.visit_expr, node.decorator_list)
        with self.new_block(node.name, node, BlockType.Class):
            self.visit_all(self.visit_stmt, node.body)

    @visit_stmt.register
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.visit_function(node)

    @visit_stmt.register
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_function(node, is_async=True)

    @visit_expr.register
    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.visit_all(
            self.visit_expr, node.args.defaults + node.args.kw_defaults
        )
        with self.new_block("lambda", node, BlockType.Function):
            self.visit_arguments(node.args)
            self.visit_expr(node.body)

    @visit_stmt.register
    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            # TO-DO(low): error handling
            self.add_definition(name, ScopeFlags.GLOBAL)

    @visit_stmt.register
    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            # TO-DO(low): error handling
            self.add_definition(name, ScopeFlags.NONLOCAL)

    @visit_expr.register(ast.Yield)
    @visit_expr.register(ast.YieldFrom)
    def visit_yield_alike(self, node: Union[ast.Yield, ast.YieldFrom]) -> None:
        if node.value is not None:
            self.visit_expr(node.value)
        self.current.set_flag("generator")

    @visit_expr.register
    def visit_Await(self, node: ast.Await) -> None:
        self.visit_expr(node.value)
        self.current.set_flag("coroutine")

    @visit_expr.register
    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            flag = ScopeFlags.USED
        elif isinstance(node.ctx, ast.Store):
            flag = ScopeFlags.LOCAL
        else:
            return None

        self.add_definition(node.id, flag)
        if (
            flag is ScopeFlags.USED
            and self.current.block_type is BlockType.Function
            and node.id == "super"
        ):
            self.add_definition("__class__", ScopeFlags.USED)

    @visit_expr.register(ast.NamedExpr)
    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if self.current.get_flag("comprehension_iterator"):
            raise CompilerError

        if self.current.get_flag("comprehension"):
            assert isinstance(target, ast.Name)
            target_name = target.id

            # Find the closest function/module block
            # and leak the target_name into there
            for entry in reversed(self.stack):
                if entry.get_flag("comprehension"):
                    if entry.lookup(target_name) & COMP_ITER:
                        raise CompilerError
                    else:
                        continue

                if entry.block_type is BlockType.Function:
                    if self.lookup(target_name, entry) & ScopeFlags.GLOBAL:
                        flag = ScopeFlags.GLOBAL
                    else:
                        flag = ScopeFlags.NONLOCAL

                    # Mark the variable in the current scope
                    # with GLOBAL/NONLOCAL
                    self.add_definition(target_name, flag)
                    # Mark the variable in the leaked scope
                    # with LOCAL
                    self.add_definition(target_name, ScopeFlags.LOCAL, entry)
                elif entry.block_type is BlockType.Module:
                    # Mark the variable in the both scopes with GLOBAL
                    self.add_definition(target_name, ScopeFlags.GLOBAL)
                    self.add_definition(target_name, ScopeFlags.GLOBA, entry)
                elif entry.block_type is BlockType.Class:
                    raise CompilerError

        self.visit(node.value)
        self.visit(node.target)

    @visit_expr.register(ast.SetComp)
    @visit_expr.register(ast.ListComp)
    @visit_expr.register(ast.GeneratorExp)
    def visit_single_element_comprehension(
        self, node: Union[ast.SetComp, ast.ListComp, ast.GeneratorExp]
    ) -> None:
        self.visit_expr(node.elt)
        self.visit_comprehensions(node)

    @visit_expr.register(ast.DictComp)
    def visit_double_element_comprehension(self, node: ast.DictComp) -> None:
        self.visit_expr(node.key)
        self.visit_expr(node.value)
        self.visit_comprehensions(node)

    def lookup(self, name: str, entry: SymbolTableEntry = None) -> FlagType:
        entry = entry or self.current
        mangled = mangle(self.namespace, name)
        return entry.symbols[mangled]

    def add_definition(
        self, name: str, flag: IntFlag, entry: SymbolTableEntry = None
    ):
        entry = entry or self.current
        mangled = mangle(self.namespace, name)

        flags = entry.symbols[mangled]
        if flags & ScopeFlags.PARAM and flag & ScopeFlags.PARAM:
            raise CompilerError

        flags |= flag
        if entry.get_flag("comprehension_target"):
            if flags & (ScopeFlags.GLOBAL | ScopeFlags.NONLOCAL):
                raise CompilerError

            flag |= ScopeFlags.COMP_ITER

        entry.symbols[mangled] = flags
        if flag & ScopeFlags.GLOBAL:
            self.global_namespace[mangled] |= flag

    def add_implicit_parameter(self, pos: int) -> None:
        # Python's comprehensions actually creates 'implicit functions' which receives
        # the outermost comprehension as the argument. These unnamed arguments should
        # not be accessible through usual syntax (but they can be through custom ASTs)
        # or like locals() calls
        self.add_definition(f".{pos}", ScopeFlags.PARAM)


@dataclass
class SymbolTableEntry:

    name: str
    node: ast.AST
    block_type: BlockType

    symbols: Symbols = _symbol_field
    analysis: Analysis = field(default_factory=dict)
    children: List[SymbolTableEntry] = field(default_factory=list)

    # generator, coroutine, comprehension etc.
    _flags: Dict[str, bool] = field(default_factory=dict)

    def get_flag(self, flag: str) -> bool:
        return self._flags.get(flag, False)

    def set_flag(self, flag: str, value: bool = True) -> None:
        self._flags[flag] = value

    @contextmanager
    def on_flag(self, flag: str) -> Generator[None, None, None]:
        try:
            self.set_flag(flag)
            yield
        finally:
            self.set_flag(flag, False)

    @property
    def parameters(self):
        return [
            symbol
            for symbol, flags in self.symbols.items()
            if flags & ScopeFlags.PARAM
        ]


@dataclass
class AnalysisState:
    free_namespace: Set[str] = field(default_factory=set)
    local_namespace: Set[str] = field(default_factory=set)
    bound_namespace: Set[str] = field(default_factory=set)
    global_namespace: Set[str] = field(default_factory=set)

    # symbol => scope mapping
    scopes: Analysis = field(default_factory=dict)

    def inherit(self, child: AnalysisState) -> None:
        child.bound_namespace |= self.bound_namespace
        child.global_namespace |= self.global_namespace
