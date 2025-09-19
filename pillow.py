#!/usr/bin/env python3

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Iterator, Union
from re import fullmatch
from subprocess import run
from sys import argv, stderr
from pathlib import Path
from struct import pack, unpack
from more_itertools import peekable
import os

class TokenType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    DEBUG = auto()
    NAME = auto()
    ROLL = auto()
    OVER = auto()
    POP = auto()
    IMPORT = auto()
    EXPORT = auto()
    PROC = auto()
    ASM = auto()
    EXTERN = auto()
    STRUCT = auto()
    ARROW = auto()
    OF = auto()
    DO = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    THEN = auto()
    END = auto()

class Token():
    def __init__(self, token_type: TokenType, value: Any = None):
        self.token_type = token_type
        self.value = value
        self.filename: str | None
        self.line_number: int | None
        self.column_number: int | None
    
    def location(self, filename: str, line_number: int, column_number: int) -> Token:
        self.filename = filename
        self.line_number = line_number
        self.column_number = column_number
        return self

    def __repr__(self) -> str:
        return self.token_type.name.lower() + (f" {self.value}" if self.value is not None else "")
        
block_stack: list[tuple[int, Token]] = []
def lex_token(tok: str, i: int) -> Token:

    try: return Token(TokenType.INTEGER, int(tok))
    except ValueError: pass

    try: return Token(TokenType.FLOAT, float(tok))
    except ValueError: pass

    if tok.startswith("\"") and tok.endswith("\""): return Token(TokenType.STRING, tok[1:-1])
    tok = tok.lower()
    comp_value = None

    m = fullmatch(r"(.*)\((.*)\)", tok)
    if m:
        tok, comp_value = m.groups()
        comp_value = str(comp_value)

    if tok == "?": return Token(TokenType.DEBUG)
    if tok == "dup": return Token(TokenType.OVER, 0)
    if tok == "swp": return Token(TokenType.ROLL, 1)
    if tok == "rot": return Token(TokenType.ROLL, 2)
    if tok == "roll": return Token(TokenType.ROLL, int(comp_value or 2))
    if tok == "over": return Token(TokenType.OVER, int(comp_value or 1))
    if tok == "pop": return Token(TokenType.POP)
    if tok == "->": return Token(TokenType.ARROW)
    if tok == "of": return Token(TokenType.OF)
    if tok == "do": return Token(TokenType.DO)
    if tok == "import": return Token(TokenType.IMPORT, comp_value)
    if tok == "export": return Token(TokenType.EXPORT)
    if tok == "proc":
        token = Token(TokenType.PROC)
        block_stack.append((i, token))
        return token
    if tok == "struct":
        token = Token(TokenType.STRUCT)
        block_stack.append((i, token))
        return token
    if tok == "asm":
        token = Token(TokenType.ASM)
        block_stack.append((i, token))
        return token
    if tok == "extern":
        token = Token(TokenType.EXTERN)
        block_stack.append((i, token))
        return token
    if tok == "if":
        token = Token(TokenType.IF)
        block_stack.append((i, token))
        return token
    if tok == "else":
        assert len(block_stack) > 0, "Else is missing corresponding if"
        opening_i, opening_tok = block_stack.pop()
        assert opening_tok.token_type == TokenType.IF, "Else is missing corresponding if"
        opening_tok.value = i
        token = Token(TokenType.ELSE)
        block_stack.append((i, token))
        return token
    if tok == "while":
        token = Token(TokenType.WHILE)
        block_stack.append((i, token))
        return token
    if tok == "then":
        assert len(block_stack) > 0, "Then is missing corresponding while"
        opening_i, opening_tok = block_stack.pop()
        assert opening_tok.token_type == TokenType.WHILE, "Then is missing corresponding while"
        token = Token(TokenType.THEN, opening_i)
        block_stack.append((i, token))
        return token
    if tok == "end":
        assert len(block_stack) > 0, "Mismatched end"
        opening_i, opening_tok = block_stack.pop()
        assert opening_tok.token_type != TokenType.WHILE, "While is missing corresponding then"
        if opening_tok.token_type == TokenType.THEN: opening_i = opening_tok.value
        opening_tok.value = i
        return Token(TokenType.END, opening_i)
    return Token(TokenType.NAME, tok)

def lex(code: str, filename: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line_number = 1
    column_start = 0
    while i < len(code):
        if code[i].isspace():
            if code[i] == "\n":
                line_number += 1
                column_start = i
            i += 1
            continue
        if code[i] == "\"":
            start = i
            i += 1
            while i < len(code) and (code[i] != "\"" or code[i - 1] == "\\"):
                i += 1
            i += 1
            tokens.append(lex_token(code[start:i], len(tokens)).location(filename, line_number, i - column_start + 1))
            continue
        if code[i] == "#":
            while i < len(code) and code[i] != "\n":
                i += 1
            continue
        start = i
        while i < len(code) and not code[i].isspace():
            if code[i] == "(":
                while i < len(code) and code[i] != ")":
                    i += 1
                assert code[i] == ")", f"Mismatched parenthesis"
            i += 1
        tokens.append(lex_token(code[start:i], len(tokens)).location(filename, line_number, i - column_start + 1))
    assert len(block_stack) == 0, f"Mismatched {block_stack[-1]}"
    return tokens

def fasm_ident_safe(s: str) -> str:
    out = []
    for ch in s:
        if fullmatch(r"[a-zA-Z_]", ch):
            out.append(ch)
        else:
            out.append(f"_u{ord(ch):04X}_")
    return "".join(out)

class Procedure():
    def __init__(self, name: str, public: bool, takes: list[PillowTypeInstance], gives: list[PillowTypeInstance], code: list[tuple[int, Token]], info: EmitInfo):
        self.name = name
        self.public = public
        self.takes = takes
        self.gives = gives
        self.code = code
        self.source_file = info.source_file
        self.proc_name = fasm_ident_safe(info.global_prefix) + "_proc_" + fasm_ident_safe(self.name) + "".join("_" + str(x) for x in self.takes)
    
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        output = ""
        def e(x: str) -> None:
            nonlocal output
            output += x + "\n"

        proc_emit_info = EmitInfo(
                                  self.source_file,
                                  info.target,
                                  False,
                                  include_intrinsics=False,
                                  type_stack=self.takes.copy(),
                                  procedures=info.procedures,
                                  types=info.types)
        assembly, data_section = emit(self.code, proc_emit_info)
        exit_stack = proc_emit_info.type_stack
        assert exit_stack == self.gives, f"Procedure {self} does not match type signature\nExpected {self.gives} but got {exit_stack}"

        e(self.proc_name + ":")
        e("push rbp")
        e("mov rbp, rsp")
        e(assembly)
        e("leave")
        e("ret")

        return output, data_section

    def call(self) -> str:
        return "call " + self.proc_name
    
    def __repr__(self) -> str:
        return f"{self.name} {self.takes} -> {self.gives}"

class AsmProcedure(Procedure):
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        return "", ""

    def call(self) -> str:
        assert all(tok.token_type == TokenType.STRING for _, tok in self.code), "Assembly procedures must have only strings in them"

        return "\n".join([tok.value for _, tok in self.code])

class ExternProcedure(Procedure):
    def __init__(self, name: str, public: bool, takes: list[PillowTypeInstance], gives: list[PillowTypeInstance], code: str, info: EmitInfo):
        self.name = name
        self.public = public
        self.takes = takes
        self.gives = gives
        self.output = code
        self.source_file = info.source_file
        self.proc_name = fasm_ident_safe(info.global_prefix) + "_proc_" + fasm_ident_safe(self.name) + "".join("_" + str(x) for x in self.takes)

    def emit(self, info: EmitInfo) -> tuple[str, str]:
        return f"extrn {self.name}\n", ""

    def call(self) -> str:
        return self.output


class StructProcedure(Procedure):
    def __init__(self, struct: PillowType, props: dict[str, PillowTypeInstance], info: EmitInfo):
        self.name = struct.name
        self.source_file = info.source_file
        self.struct = struct
        self.public = struct.public
        self.takes: list[PillowTypeInstance] = list(props.values())
        self.gives: list[PillowTypeInstance] = [struct()]
        self.proc_name = fasm_ident_safe(info.global_prefix) + "_proc_" + fasm_ident_safe(self.name) + "".join("_" + str(x) for x in self.takes)
    
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        output = ""
        def e(x: str) -> None:
            nonlocal output
            output += x + "\n"

        e(self.proc_name + ":")
        e("push rbp")
        e("mov rbp, rsp")
        match info.target:
            case Target.LINUX:
                e("mov rdi, " + str(self.struct.size))
                e("call malloc")
            case Target.WINDOWS:
                e("sub rsp, 0x20")
                e("mov rcx, " + str(self.struct.size))
                e("call [malloc]")
        for i in reversed(range(0, len(self.takes))):
            e("spop rbx")
            e("mov [rax+" + str(i*8) + "], rbx")
        e("spush rax")
        e("leave")
        e("ret")

        return output, ""

class StructPropProcedure(Procedure):
    def __init__(self, struct: PillowType, prop: str, prop_type: PillowTypeInstance, offset: int, info: EmitInfo):
        self.name = "."+prop
        self.source_file = info.source_file
        self.offset = offset
        self.public = struct.public
        self.prop = prop
        self.takes: list[PillowTypeInstance] = [struct()]
        self.gives: list[PillowTypeInstance] = [prop_type]
        self.proc_name = fasm_ident_safe(info.global_prefix) + "_proc_" + fasm_ident_safe(self.name) + "".join("_" + str(x) for x in self.takes)
    
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        output = ""
        def e(x: str) -> None:
            nonlocal output
            output += x + "\n"

        e(self.proc_name + ":")
        e("push rbp")
        e("mov rbp, rsp")
        e("spop rax")
        e("mov rbx, [rax+" + str(self.offset) + "]")
        e("spush rbx")
        e("leave")
        e("ret")

        return output, ""

class PillowType():
    def __init__(self, name: str, public: bool, size: int, children_count: int = 0):
        self.name = name
        self.public = public
        self.size = size
        self.children_count = children_count

    def __call__(self, *children: PillowTypeInstance) -> PillowTypeInstance:
        return PillowTypeInstance(self, list(children))

    def __repr__(self) -> str:
        return self.name.lower()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PillowType):
            return NotImplemented
        return self.name == other.name

class PillowTypeInstance():
    def __init__(self, kind: PillowType, children: list[PillowTypeInstance]):
        self.kind = kind
        assert self.kind.children_count == len(children), f"Type {self.kind} expected {self.kind.children_count} children but found {len(children)}"
        self.children = children

    def __repr__(self) -> str:
        return f"{self.kind}{" ".join(str(x) for x in self.children)}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PillowTypeInstance):
            return NotImplemented
        return self.kind == other.kind and self.children == other.children

class PillowPrimitive:
    INT = PillowType("int", True, 8)
    FLO = PillowType("flo", True, 8)
    CHR = PillowType("chr", True, 1)
    PTR = PillowType("ptr", True, 8, 1)

class Target(Enum):
    LINUX = auto()
    WINDOWS = auto()

class EmitInfo():
    def __init__(self,
                 source_file: str,
                 target: Target,
                 binary: bool = False,
                 global_prefix: str = "",
                 include_intrinsics: bool = True,
                 type_stack: list[PillowTypeInstance] | None = None,
                 procedures: list[Procedure] | None = None,
                 types: list[PillowType] | None = None,
                 imports: list[str] | None = None,
                 ):
        self.source_file = source_file
        self.target = target
        self.binary = binary
        self.global_prefix = global_prefix
        self.include_intrinsics = include_intrinsics
        self.type_stack = type_stack or []
        self.procedures = procedures or []
        self.types = types or [PillowPrimitive.INT, PillowPrimitive.FLO, PillowPrimitive.CHR, PillowPrimitive.PTR]
        self.imports = imports or []

def parse_type(code: Iterator[tuple[int, Token]], info: EmitInfo) -> PillowTypeInstance:
    _, type_name = next(code)
    assert type_name.token_type == TokenType.NAME, f"Expected type name but found {type_name}"

    kind = next((kind for kind in info.types if kind.name == type_name.value), None)
    assert kind is not None, f"Undefined type {type_name.value}"

    if kind.children_count == 0: return kind()

    _, of_token = next(code)
    assert of_token.token_type == TokenType.OF, f"Expected of after type {kind}"

    children_types: list[PillowTypeInstance] = []
    for _ in range(kind.children_count):
        children_types.append(parse_type(code, info))

    return kind(*children_types)

def parse_parameters(code: peekable[tuple[int, Token]], info: EmitInfo) -> tuple[list[PillowTypeInstance], list[PillowTypeInstance]]:
    takes_type = None
    takes_types: list[PillowTypeInstance] = []
    while True:
        _, takes_token = code.peek()
        if takes_token.token_type == TokenType.NAME: takes_types.append(parse_type(code, info))
        else:
            assert takes_token.token_type == TokenType.ARROW, f"Expected type or arrow but found {takes_type}"
            break
    next(code)
    gives_type = None
    gives_types: list[PillowTypeInstance] = []
    while True:
        _, gives_token = code.peek()
        if gives_token.token_type == TokenType.NAME: gives_types.append(parse_type(code, info))
        else: break
    return takes_types, gives_types


def emit(code: list[tuple[int, Token]], info: EmitInfo) -> tuple[str, str]:
    output = ""
    data_section = ""

    def e(x: str) -> None:
        nonlocal output
        output += x + "\n"

    def d(x: str) -> None:
        nonlocal data_section
        data_section += x + "\n"

    def type_stack_op(tok: Token, takes: list[PillowTypeInstance], gives: list[PillowTypeInstance]) -> None:
        nonlocal info
        if len(takes) > 0:
            assert info.type_stack[-len(takes):] == takes, f"Instruction {tok} takes {takes} but found {info.type_stack[-len(takes):]}"
            info.type_stack = info.type_stack[:-len(takes)]
        info.type_stack += gives


    if info.binary:
        if info.target == Target.LINUX:
            e("TARGET = 0")
        else:
            e("TARGET = 1")
        e("include 'components/header.inc'")

    if info.include_intrinsics:
        with open("./packages/intrinsics.pilo", "r") as f:
            assert not f.closed, f"Could not import intrinsics"
            intrinsics_file_path = os.path.abspath("./packages/intrinsics.pilo")
            intrinsics_emit_info = EmitInfo(intrinsics_file_path, info.target, False, include_intrinsics=False)
            intrinsics_assembly, intrinsics_data_section = emit(list(enumerate(lex(f.read(), intrinsics_file_path))), intrinsics_emit_info)
            info.procedures += [p for p in intrinsics_emit_info.procedures if p.source_file not in info.imports]
            info.types += intrinsics_emit_info.types
            info.imports += intrinsics_emit_info.imports
            info.imports.append(intrinsics_file_path)
            if info.binary: d(intrinsics_data_section)

    block_type_stack: list[list[PillowTypeInstance]] = []

    code_iter = peekable(iter(code))

    next_public = False
    for i, tok in code_iter:
        try:
            def t(takes: list[PillowTypeInstance], gives: list[PillowTypeInstance]) -> None:
                type_stack_op(tok, takes, gives)
            
            match tok.token_type:
                case TokenType.INTEGER:
                    t([], [PillowPrimitive.INT()])
                    e("spush " + str(tok.value))
                case TokenType.FLOAT:
                    t([], [PillowPrimitive.FLO()])
                    bits, = unpack("<Q", pack("<d", tok.value))
                    e("spush " + hex(bits))
                case TokenType.STRING:
                    t([], [PillowPrimitive.PTR(PillowPrimitive.CHR())])
                    d(info.global_prefix + "string_" + str(i) + " db \"" + tok.value + "\", 0")
                    e("spush " + info.global_prefix + "string_" + str(i))
                case TokenType.DEBUG:
                    print(info.type_stack)
                case TokenType.ROLL:
                    roll_size = tok.value + 1
                    assert len(info.type_stack) >= roll_size, f"Instruction {tok} takes {roll_size} items but found {len(info.type_stack)}"
                    takes = info.type_stack[-roll_size:]
                    gives = [*takes[1:], takes[0]]
                    t(takes, gives)
                    for _ in range(roll_size-1):
                        e("spop rax")
                        e("push rax")
                    e("spop rbx")
                    for _ in range(roll_size-1):
                        e("pop rax")
                        e("spush rax")
                    e("spush rbx")
                case TokenType.OVER:
                    over_size = tok.value + 1
                    assert len(info.type_stack) >= over_size, f"Instruction {tok} takes {over_size} items but found {len(info.type_stack)}"
                    takes = info.type_stack[-over_size:]
                    gives = [*takes, takes[0]]
                    t(takes, gives)
                    e("mov rax, [r12+" + str(8*(over_size-1)) + "]")
                    e("spush rax")
                case TokenType.POP:
                    assert len(info.type_stack) >= 1, f"Instruction {tok} takes 1 item but found {len(info.type_stack)}"
                    t([info.type_stack[-1]], [])
                    e("add r12, 8")
                case TokenType.NAME:
                    assert tok.value in [procedure.name for procedure in info.procedures if procedure.public or procedure.source_file == info.source_file], f"Undefined procedure {tok.value}"
                    procedure = next((x for x in info.procedures if (x.public or x.source_file == info.source_file) and x.name == tok.value and (info.type_stack[-len(x.takes):] == x.takes or len(x.takes) == 0)), None)
                    assert procedure is not None, f"No overload for {tok.value} matches {info.type_stack}"
                    t(procedure.takes, procedure.gives)
                    e(procedure.call())
                case TokenType.IMPORT:
                    with open(tok.value, "r") as f:
                        assert not f.closed, f"Could not import file {tok.value}"
                        abs_path = os.path.abspath(tok.value)
                        import_emit_info = EmitInfo(abs_path, info.target, False, tok.value)
                        import_assembly, import_data_section = emit(list(enumerate(lex(f.read(), abs_path))), import_emit_info)
                        info.procedures += [p for p in import_emit_info.procedures if p.source_file not in info.imports]
                        info.types += import_emit_info.types
                        info.imports += import_emit_info.imports
                        info.imports.append(os.path.abspath(tok.value))
                        d(import_data_section)
                case TokenType.EXPORT:
                    assert code[i+1][1].token_type in [TokenType.PROC, TokenType.ASM, TokenType.EXTERN, TokenType.STRUCT], f"Export must be followed by proc, asm, or struct"
                    next_public = True
                case TokenType.PROC | TokenType.ASM:
                    _, proc_name = next(code_iter)
                    assert proc_name.token_type == TokenType.NAME, f"Expected procedure name but found {proc_name}"
                    takes_types, gives_types = parse_parameters(code_iter, info)
                    already_defined_proc = next((procedure for procedure in info.procedures if procedure.name == proc_name.value and procedure.takes == takes_types and procedure.gives == gives_types), None)
                    assert already_defined_proc is None, f"Procedure {already_defined_proc} is already defined"
                    overload_different_takes = next((procedure for procedure in info.procedures if procedure.name == proc_name.value and len(procedure.takes) != len(takes_types)), None)
                    assert overload_different_takes is None, f"Procedure {overload_different_takes} takes a different number of items"
                    do_i, do_token = next(code_iter)
                    assert do_token.token_type == TokenType.DO, f"Expected do after parameters but found {do_token}"
                    if tok.token_type == TokenType.PROC:
                        info.procedures.append(Procedure(proc_name.value, next_public, takes_types, gives_types, code[do_i + 1:tok.value], info))
                    else:
                        info.procedures.append(AsmProcedure(proc_name.value, next_public, takes_types, gives_types, code[do_i + 1:tok.value], info))
                    while do_i < tok.value: do_i, _ = next(code_iter)
                    next_public = False
                case TokenType.EXTERN:
                    _, extern_name = next(code_iter)
                    assert extern_name.token_type == TokenType.NAME, f"Expected extern name but found {extern_name}"
                    already_defined_extern = next((kind for kind in info.types if kind.name == extern_name.value), None)
                    assert already_defined_extern is None, f"Procedure {extern_name.value} is already defined"
                    takes_types, gives_types = parse_parameters(code_iter, info)
                    _, end_token = next(code_iter)
                    assert end_token.token_type == TokenType.END, f"Expected end after parameters but found {end_token}"
                    assert len(gives_types) <= 1, f"Extern procedures can only have one return value"
                    param_registers = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
                    return_register = "rax"
                    flo_param_registers = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"]
                    flo_return_register = "xmm0"
                    assembly = ""
                    for takes_type in takes_types:
                        if takes_type.kind == PillowPrimitive.FLO:
                            assembly = f"spop {flo_param_registers.pop(0)}\n" + assembly
                        else:
                            assembly = f"spop {param_registers.pop(0)}\n" + assembly
                    assembly += f"call {extern_name.value}\n"
                    if len(gives_types) > 0:
                        if gives_types[0].kind == PillowPrimitive.FLO:
                            assembly += f"spush {flo_return_register}\n"
                        else:
                            assembly += f"spush {return_register}\n"
                    info.procedures.append(ExternProcedure(extern_name.value, next_public, takes_types, gives_types, assembly, info))
                    next_public = False
                case TokenType.STRUCT:
                    _, struct_name = next(code_iter)
                    assert struct_name.token_type == TokenType.NAME, f"Expected struct name but found {struct_name}"
                    already_defined_struct = next((kind for kind in info.types if kind.name == struct_name.value), None)
                    assert already_defined_struct is None, f"Struct {struct_name.value} is already defined"
                    props: dict[str, PillowTypeInstance] = {}
                    new_struct = PillowType(struct_name.value, next_public, 8)
                    prop_offset = 0
                    while code_iter.peek()[0] < tok.value:
                        _, prop_name_tok = next(code_iter)
                        assert prop_name_tok.token_type == TokenType.NAME, f"Expected property name but found {prop_name_tok}"

                        prop_name = prop_name_tok.value
                        assert prop_name not in props, f"Struct already has property {prop_name}"

                        props[prop_name] = parse_type(code_iter, info)

                        info.procedures.append(StructPropProcedure(new_struct, prop_name, props[prop_name], prop_offset, info))
                        prop_offset += 8
                    next(code_iter)
                    info.types.append(new_struct)
                    info.procedures.append(StructProcedure(new_struct, props, info))
                    next_public = False
                case TokenType.IF:
                    t([PillowPrimitive.INT()], [])
                    block_type_stack.append(info.type_stack.copy())
                    e("spop rax")
                    e("test rax, rax")
                    e("jz " + fasm_ident_safe(info.source_file) + "_label_" + str(tok.value))
                case TokenType.ELSE:
                    old_block_type_stack = block_type_stack.pop()
                    block_type_stack.append(info.type_stack.copy())
                    info.type_stack = old_block_type_stack
                    e("jmp " + fasm_ident_safe(info.source_file) + "_label_" + str(tok.value))
                    e(fasm_ident_safe(info.source_file) + "_label_" + str(i) + ":")
                case TokenType.WHILE:
                    block_type_stack.append(info.type_stack.copy())
                    e(fasm_ident_safe(info.source_file) + "_label_" + str(i) + ":")
                case TokenType.THEN:
                    t([PillowPrimitive.INT()], [])
                    old_block_type_stack = block_type_stack.pop()
                    block_type_stack.append(info.type_stack.copy())
                    info.type_stack = old_block_type_stack
                    e("spop rax")
                    e("test rax, rax")
                    e("jz " + fasm_ident_safe(info.source_file) + "_label_" + str(tok.value))
                case TokenType.END:
                    _, opening_token = next(x for x in code if x[0] == tok.value)
                    assert len(block_type_stack) != 0, f"Mismatched end"
                    old_block_type_stack = block_type_stack.pop()
                    match opening_token.token_type:
                        case TokenType.IF:
                            assert old_block_type_stack == info.type_stack, f"If statement must leave the stack the same as before\nStarted with {old_block_type_stack} and got {info.type_stack}"
                        case TokenType.ELSE:
                            assert old_block_type_stack == info.type_stack, f"If else statement must leave the stack the same in both branches\nIf branch got {old_block_type_stack}, else branch got {info.type_stack}"
                        case TokenType.WHILE:
                            assert old_block_type_stack == info.type_stack, f"While loop must leave the stack the same as before\nStarted with {old_block_type_stack} and got {info.type_stack}"
                            e("jmp " + fasm_ident_safe(info.source_file) + "_label_" + str(tok.value))
                    e(fasm_ident_safe(info.source_file) + "_label_" + str(i) + ":")
                case _:
                    raise Exception(f"Token type {tok.token_type} not implemented")
        except AssertionError as exception:
            print(f"Error at {tok.filename}:{tok.line_number}:{tok.column_number}: {exception}", file=stderr)
            exit(1)

    if info.binary:
        e("jmp pillow_exit")

        for procedure in info.procedures:
            proc_code, proc_data = procedure.emit(info)
            e(proc_code)
            d(proc_data)

        e("include 'components/footer.inc'")
        e(data_section)

    return output, data_section

def compile(assembly: str, target: Target, output_path: Path) -> None:
    with open(output_path.with_suffix(".s"), "w") as f:
        f.write(assembly)
    fasm_result = run([".\\fasm2\\fasm2.cmd" if target == Target.WINDOWS else "./fasm2/fasm2", "-n", output_path.with_suffix(".s")])
    assert fasm_result.returncode == 0, "Failed to assemble"
    if target == Target.LINUX:
        run(["clang", "-no-pie", output_path.with_suffix(".o"), "-o", output_path.with_suffix("")])

def usage() -> None:
    print("Usage: ./pillow.py inputfile [outputfile]")
    exit(1)

def main() -> None:
    if len(argv) < 2: usage()

    target = Target.WINDOWS if os.name == "nt" else Target.LINUX
    output_file = Path(argv[-1])

    with open(argv[1], "r") as f:
        if f.closed:
            print(f"Could not open file {argv[-1]} for reading")
            exit(1)
        abs_path = os.path.abspath(argv[-1])
        assembly, _ = emit(list(enumerate(lex(f.read(), abs_path))), EmitInfo(abs_path, target, True, procedures=[]))
        compile(assembly, target, output_file)


if __name__ == "__main__":
    main()
