#!/usr/bin/env python3

from __future__ import annotations
from enum import Enum, auto
from typing import Any, Union
from re import fullmatch
from subprocess import run
from sys import argv
from pathlib import Path
from struct import pack, unpack
import os

class TokenType(Enum):
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    NAME = auto()
    DUP = auto()
    SWP = auto()
    ROLL = auto()
    OVER = auto()
    POP = auto()
    IMPORT = auto()
    EXPORT = auto()
    PROC = auto()
    ASM = auto()
    STRUCT = auto()
    INT_TYPE = auto()
    FLO_TYPE = auto()
    STR_TYPE = auto()
    ARROW = auto()
    DO = auto()
    DUMP = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    THEN = auto()
    END = auto()

primitives = [TokenType.INT_TYPE, TokenType.FLO_TYPE, TokenType.STR_TYPE]

class Token():
    def __init__(self, token_type: TokenType, value: Any = None):
        self.token_type = token_type
        self.value = value

    def __repr__(self) -> str:
        return self.token_type.name.lower() + (f" {self.value}" if self.value is not None else "")
        
class PillowType(Enum):
    INT = auto()
    FLO = auto()
    STR = auto()
    
    def __repr__(self) -> str:
        return self.name.lower()

    def __str__(self) -> str:
        return self.name.lower()

block_stack: list[tuple[int, Token]] = []
def lex_token(tok: str, i: int) -> Token:
    try:
        return Token(TokenType.INTEGER, int(tok))
    except ValueError:
        pass
    try:
        return Token(TokenType.FLOAT, float(tok))
    except ValueError:
        pass
    if tok.startswith("\"") and tok.endswith("\""): return Token(TokenType.STRING, tok[1:-1])
    tok = tok.lower()
    comp_value = None

    m = fullmatch(r"(.*)\((.*)\)", tok)
    if m:
        tok, comp_value = m.groups()
        comp_value = str(comp_value)

    if tok == "dup": return Token(TokenType.DUP)
    if tok == "swp": return Token(TokenType.SWP)
    if tok == "rot": return Token(TokenType.ROLL, 2)
    if tok == "roll": return Token(TokenType.ROLL, int(comp_value or 2))
    if tok == "over": return Token(TokenType.OVER, int(comp_value or 1))
    if tok == "pop": return Token(TokenType.POP)
    if tok == "dump": return Token(TokenType.DUMP)
    if tok == "int": return Token(TokenType.INT_TYPE)
    if tok == "flo": return Token(TokenType.FLO_TYPE)
    if tok == "str": return Token(TokenType.STR_TYPE)
    if tok == "->": return Token(TokenType.ARROW)
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

def lex(code: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(code):
        if code[i].isspace():
            i += 1
            continue
        if code[i] == "\"":
            start = i
            i += 1
            while i < len(code) and (code[i] != "\"" or code[i - 1] == "\\"):
                i += 1
            i += 1
            tokens.append(lex_token(code[start:i], len(tokens)))
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
        tokens.append(lex_token(code[start:i], len(tokens)))
    assert len(block_stack) == 0, f"Mismatched {block_stack[-1]}"
    return tokens

type GlobalProc = Procedure
type GlobalType = Union[PillowType, Struct]

def fasm_ident_safe(s: str) -> str:
    out = []
    for ch in s:
        if fullmatch(r"[a-zA-Z_]", ch):
            out.append(ch)
        else:
            out.append(f"_u{ord(ch):04X}_")
    return "".join(out)

class Procedure():
    def __init__(self, name: str, public: bool, takes: list[GlobalType], gives: list[GlobalType], code: list[tuple[int, Token]], info: EmitInfo):
        self.name = name
        self.public = public
        self.takes = takes
        self.gives = gives
        self.code = code
        self.proc_name = info.global_prefix + "_proc_" + fasm_ident_safe(self.name) + "".join("_" + str(x) for x in self.takes)
    
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        output = ""
        def e(x: str) -> None:
            nonlocal output
            output += x + "\n"

        proc_emit_info = EmitInfo(info.target,
                                  False,
                                  include_intrinsics=False,
                                  type_stack=self.takes.copy(),
                                  procedures=info.procedures,
                                  structs=info.structs)
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

class StructProcedure(Procedure):
    def __init__(self, struct: Struct, info: EmitInfo):
        self.name = struct.name
        self.struct = struct
        self.public = struct.public
        self.takes: list[GlobalType] = list(struct.props.values())
        self.gives: list[GlobalType] = [struct]
        self.proc_name = info.global_prefix + "_proc_" + self.name + "".join("_" + str(x) for x in self.takes)
    
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
                e("mov rdi, " + str(self.struct.size()))
                e("call malloc")
            case Target.WINDOWS:
                e("sub rsp, 0x20")
                e("mov rcx, " + str(self.struct.size()))
                e("call [malloc]")
        for i in reversed(range(0, len(self.takes))):
            e("spop rbx")
            e("mov [rax+" + str(i*8) + "], rbx")
        e("spush rax")
        e("leave")
        e("ret")

        return output, ""

class StructPropProcedure(Procedure):
    def __init__(self, struct: Struct, prop: str, info: EmitInfo):
        self.name = "."+prop
        self.struct = struct
        self.public = struct.public
        self.prop = prop
        self.takes: list[GlobalType] = [struct]
        self.gives: list[GlobalType] = [struct.props[prop]]
        self.proc_name = info.global_prefix + "_proc_" + self.name + "".join("_" + str(x) for x in self.takes)
    
    def emit(self, info: EmitInfo) -> tuple[str, str]:
        output = ""
        def e(x: str) -> None:
            nonlocal output
            output += x + "\n"

        e(self.proc_name + ":")
        e("push rbp")
        e("mov rbp, rsp")
        e("spop rax")
        prop_index = list(self.struct.props.keys()).index(self.prop)
        e("mov rbx, [rax+" + str(prop_index*8) + "]")
        e("spush rbx")
        e("leave")
        e("ret")

        return output, ""

class Struct():
    def __init__(self, name: str, public: bool, code: list[tuple[int, Token]], structs: list[Struct]):
        self.name = name
        self.public = public
        self.props: dict[str, GlobalType] = {}

        toks = [tok for _, tok in code]
        pairs = list(zip(toks[::2], toks[1::2]))
        assert len(pairs) % 2 == 0, f"Structs must have name type pairs"
        for prop_name_tok, prop_type_tok in pairs:
            assert prop_name_tok.token_type == TokenType.NAME, f"Expected property name but found {prop_name_tok}"
            assert prop_type_tok.token_type in [*primitives, TokenType.NAME], f"Expected property type but found {prop_type_tok}"

            prop_name = prop_name_tok.value
            assert prop_name not in self.props, f"Struct already has property {prop_name}"

            prop_type: GlobalType
            match prop_type_tok.token_type:
                case TokenType.INT_TYPE:
                    prop_type = PillowType.INT
                case TokenType.FLO_TYPE:
                    prop_type = PillowType.FLO
                case TokenType.STR_TYPE:
                    prop_type = PillowType.STR
                case TokenType.NAME:
                    prop_struct = next((struct for struct in structs if struct.name == prop_type_tok.value), None)
                    assert prop_struct is not None, f"Undefined type {prop_type_tok.value}"
                    prop_type = prop_struct

            self.props[prop_name] = prop_type

    def size(self) -> int:
        return len(self.props)*8

    def __repr__(self) -> str:
        return f"{self.name}"

class Target(Enum):
    LINUX = auto()
    WINDOWS = auto()

class EmitInfo():
    def __init__(self,
                 target: Target,
                 binary: bool = False,
                 global_prefix: str = "",
                 include_intrinsics: bool = True,
                 type_stack: list[GlobalType] | None = None,
                 procedures: list[GlobalProc] | None = None,
                 structs: list[Struct] | None = None,
                 ):
        self.target = target
        self.binary = binary
        self.global_prefix = global_prefix
        self.include_intrinsics = include_intrinsics
        self.type_stack = type_stack or []
        self.procedures = procedures or []
        self.structs = structs or []

def emit(code: list[tuple[int, Token]], info: EmitInfo) -> tuple[str, str]:
    output = ""
    data_section = ""

    def e(x: str) -> None:
        nonlocal output
        output += x + "\n"

    def d(x: str) -> None:
        nonlocal data_section
        data_section += x + "\n"

    def type_stack_op(tok: Token, takes: list[GlobalType], gives: list[GlobalType]) -> None:
        nonlocal info
        if len(takes) > 0:
            assert info.type_stack[-len(takes):] == takes, f"Instruction {tok} takes {takes} but found {info.type_stack[-len(takes):]}"
            info.type_stack = info.type_stack[:-len(takes)]
        info.type_stack += gives


    if info.binary:
        match info.target:
            case Target.LINUX:
                e("format ELF64")

                e("public main")
                e("extrn exit")
                e("extrn printf")
                e("extrn malloc")
                e("section '.text' executable")

                e("macro spush value")
                e("    sub r12, 8")
                e("    mov r8, value")
                e("    mov qword [r12], r8")
                e("end macro")

                e("macro spop value")
                e("    mov value, qword [r12]")
                e("    add r12, 8")
                e("end macro")

                e("macro spushsd xmmreg")
                e("    sub r12, 8")
                e("    movq [r12], xmmreg")
                e("end macro")

                e("macro spopsd xmmreg")
                e("    movq xmmreg, [r12]")
                e("    add r12, 8")
                e("end macro")

                e("main:")
                e("lea r12, [pillow_stack + 4096]")

            case Target.WINDOWS:
                e("format PE64")
                e("entry main")

                e("include 'win64a.inc'")
                e("section '.text' code executable")

                e("macro spush value")
                e("    sub r12, 8")
                e("    mov rax, value")
                e("    mov qword [r12], rax")
                e("end macro")

                e("macro spop value")
                e("    mov value, qword [r12]")
                e("    add r12, 8")
                e("end macro")

                e("macro spushsd xmmreg")
                e("    sub r12, 8")
                e("    movq [r12], xmmreg")
                e("end macro")

                e("macro spopsd xmmreg")
                e("    movq xmmreg, [r12]")
                e("    add r12, 8")
                e("end macro")

                e("main:")
                e("lea r12, [pillow_stack + 4096]")

    if info.include_intrinsics:
        with open("./packages/intrinsics.pilo", "r") as f:
            assert not f.closed, f"Could not import intrinsics"
            intrinsics_emit_info = EmitInfo(info.target, False, include_intrinsics=False)
            intrinsics_assembly, intrinsics_data_section = emit(list(enumerate(lex(f.read()))), intrinsics_emit_info)
            info.procedures += [p for p in intrinsics_emit_info.procedures if p.public]
            info.structs += [s for s in intrinsics_emit_info.structs if s.public]
            if info.binary: d(intrinsics_data_section)

    block_type_stack: list[list[GlobalType]] = []

    code_iter = iter(code)

    next_public = False

    for i, tok in code_iter:

        def t(takes: list[GlobalType], gives: list[GlobalType]) -> None:
            type_stack_op(tok, takes, gives)
        
        match tok.token_type:
            case TokenType.INTEGER:
                t([], [PillowType.INT])
                e("spush " + str(tok.value))
            case TokenType.FLOAT:
                t([], [PillowType.FLO])
                bits, = unpack("<Q", pack("<d", tok.value))
                e("spush " + hex(bits))
            case TokenType.STRING:
                t([], [PillowType.STR])
                d(info.global_prefix + "string_" + str(i) + " db \"" + tok.value + "\", 0")
                e("spush " + info.global_prefix + "string_" + str(i))
            case TokenType.DUP:
                info.type_stack.append(info.type_stack[-1])
                e("mov rax, [r12]")
                e("spush rax")
            case TokenType.SWP:
                assert len(info.type_stack) >= 2, f"Instruction {tok} takes 2 items but found {len(info.type_stack)}"
                takes = info.type_stack[-2:]
                gives = [takes[1], takes[0]]
                t(takes, gives)
                e("spop rax")
                e("spop rbx")
                e("spush rax")
                e("spush rbx")
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
                gives = [*takes, takes[-1]]
                t(takes, gives)
                e("mov rax, [r12+" + str(8*(over_size-1)) + "]")
                e("spush rax")
            case TokenType.POP:
                assert len(info.type_stack) >= 1, f"Instruction {tok} takes 1 item but found {len(info.type_stack)}"
                t([info.type_stack[-1]], [])
                e("add r12, 8")
            case TokenType.DUMP:
                stack_size = len(info.type_stack)
                for ind in range(stack_size):
                    offset = (stack_size - 1 - ind) * 8
                    item_type = info.type_stack[ind]
                    match item_type:
                        case PillowType.INT:
                            e("mov rax, [r12+" + str(offset) + "]")
                            e("call print_int")
                        case PillowType.FLO:
                            e("movq xmm0, [r12+" + str(offset) + "]")
                            e("call print_flo")
                        case PillowType.STR:
                            e("mov rax, [r12+" + str(offset) + "]")
                            e("call print_str")
                    e("call print_spc")
                e("call print_ln")
            case TokenType.NAME:
                assert tok.value in [procedure.name for procedure in info.procedures], f"Undefined procedure {tok.value}"
                procedure = next((x for x in info.procedures if x.name == tok.value and (info.type_stack[-len(x.takes):] == x.takes or len(x.takes) == 0)), None)
                assert procedure is not None, f"No overload for {tok.value} matches {info.type_stack}"
                t(procedure.takes, procedure.gives)
                e(procedure.call())
            case TokenType.IMPORT:
                with open(tok.value, "r") as f:
                    assert not f.closed, f"Could not import file {tok.value}"
                    import_emit_info = EmitInfo(info.target, False, tok.value)
                    import_assembly, import_data_section = emit(list(enumerate(lex(f.read()))), import_emit_info)
                    info.procedures += [p for p in import_emit_info.procedures if p.public]
                    info.structs += [s for s in import_emit_info.structs if s.public]
                    d(import_data_section)
            case TokenType.EXPORT:
                assert code[i+1][1].token_type in [TokenType.PROC, TokenType.ASM, TokenType.STRUCT], f"Export must be followed by proc, asm, or struct"
                next_public = True
            case TokenType.PROC | TokenType.ASM:
                _, proc_name = next(code_iter)
                assert proc_name.token_type == TokenType.NAME, f"Expected procedure name but found {proc_name}"
                takes_type = None
                takes_types: list[GlobalType] = []
                while True:
                    _, takes_type = next(code_iter)
                    match takes_type.token_type:
                        case TokenType.INT_TYPE: takes_types.append(PillowType.INT)
                        case TokenType.FLO_TYPE: takes_types.append(PillowType.FLO)
                        case TokenType.STR_TYPE: takes_types.append(PillowType.STR)
                        case TokenType.NAME:
                            struct = next((x for x in info.structs if x.name == takes_type.value), None)
                            assert struct is not None, f"Undefined struct {takes_type.value}"
                            takes_types.append(struct)
                        case TokenType.ARROW: break
                        case _: assert False, f"Expected type or arrow but found {takes_type}"
                gives_type = None
                gives_types: list[GlobalType] = []
                do_i = 0
                while True:
                    do_i, gives_type = next(code_iter)
                    match gives_type.token_type:
                        case TokenType.INT_TYPE: gives_types.append(PillowType.INT)
                        case TokenType.FLO_TYPE: gives_types.append(PillowType.FLO)
                        case TokenType.STR_TYPE: gives_types.append(PillowType.STR)
                        case TokenType.NAME:
                            struct = next((x for x in info.structs if x.name == gives_type.value), None)
                            assert struct is not None, f"Undefined struct {gives_type.value}"
                            gives_types.append(struct)
                        case TokenType.DO: break
                        case _: assert False, f"Expected type or do but found {gives_type}"
                already_defined_proc = next((procedure for procedure in info.procedures if procedure.name == proc_name.value and procedure.takes == takes_types and procedure.gives == gives_types), None)
                if already_defined_proc is not None: assert False, f"Procedure {already_defined_proc} is already defined"
                overload_different_takes = next((procedure for procedure in info.procedures if procedure.name == proc_name.value and len(procedure.takes) != len(takes_types)), None)
                if overload_different_takes is not None: assert False, f"Procedure {overload_different_takes} takes a different number of items"
                if tok.token_type == TokenType.PROC:
                    info.procedures.append(Procedure(proc_name.value, next_public, takes_types, gives_types, code[do_i + 1:tok.value], info))
                else:
                    info.procedures.append(AsmProcedure(proc_name.value, next_public, takes_types, gives_types, code[do_i + 1:tok.value], info))
                while do_i < tok.value: do_i, _ = next(code_iter)
                next_public = False
            case TokenType.STRUCT:
                name_i, struct_name = next(code_iter)
                assert struct_name.token_type == TokenType.NAME, f"Expected struct name but found {struct_name}"
                props: dict[str, PillowType] = {}
                already_defined_struct = next((struct for struct in info.structs if struct.name == struct_name.value), None)
                new_struct = Struct(struct_name.value, next_public, code[name_i + 1:tok.value], info.structs)
                info.structs.append(new_struct)
                info.procedures.append(StructProcedure(new_struct, info))
                for prop_name, prop_type in new_struct.props.items():
                    info.procedures.append(StructPropProcedure(new_struct, prop_name, info))
                while name_i < tok.value: name_i, _ = next(code_iter)
                next_public = False
            case TokenType.IF:
                t([PillowType.INT], [])
                block_type_stack.append(info.type_stack.copy())
                e("spop rax")
                e("test rax, rax")
                e("jz label_" + str(tok.value))
            case TokenType.ELSE:
                old_block_type_stack = block_type_stack.pop()
                block_type_stack.append(info.type_stack.copy())
                info.type_stack = old_block_type_stack
                e("jmp label_" + str(tok.value))
                e("label_" + str(i) + ":")
            case TokenType.WHILE:
                block_type_stack.append(info.type_stack.copy())
                e("label_" + str(i) + ":")
            case TokenType.THEN:
                t([PillowType.INT], [])
                old_block_type_stack = block_type_stack.pop()
                block_type_stack.append(info.type_stack.copy())
                info.type_stack = old_block_type_stack
                e("spop rax")
                e("test rax, rax")
                e("jz label_" + str(tok.value))
            case TokenType.END:
                _, opening_token = next(x for x in code if x[0] == tok.value)
                old_block_type_stack = block_type_stack.pop()
                match opening_token.token_type:
                    case TokenType.IF:
                        assert old_block_type_stack == info.type_stack, f"If statement must leave the stack the same as before\nStarted with {old_block_type_stack} and got {info.type_stack}"
                    case TokenType.ELSE:
                        assert old_block_type_stack == info.type_stack, f"If else statement must leave the stack the same in both branches\nIf branch got {old_block_type_stack}, else branch got {info.type_stack}"
                    case TokenType.WHILE:
                        assert old_block_type_stack == info.type_stack, f"While loop must leave the stack the same as before\nStarted with {old_block_type_stack} and got {info.type_stack}"
                        e("jmp label_" + str(tok.value))
                e("label_" + str(i) + ":")
            case _:
                raise Exception(f"Token type {tok.token_type} not implemented")


    if info.binary:
        match info.target:
            case Target.LINUX:
                e("xor rax, rax")
                e("call exit")

                e("print_int:")
                e("push rbp")
                e("mov rbp, rsp")
                e("mov rdi, int_fmt")
                e("mov rsi, rax")
                e("xor rax, rax")
                e("call printf")
                e("leave")
                e("ret")

                e("print_flo:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 8")
                e("lea rdi, [flo_fmt]")
                e("mov eax, 1")
                e("call printf")
                e("leave")
                e("ret")

                e("print_str:")
                e("push rbp")
                e("mov rbp, rsp")
                e("mov rdi, str_fmt")
                e("mov rsi, rax")
                e("xor rax, rax")
                e("call printf")
                e("leave")
                e("ret")

                e("print_spc:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 32")
                e("mov rdi, space")
                e("xor rax, rax")
                e("call printf")
                e("add rsp, 32")
                e("leave")
                e("ret")

                e("print_ln:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 32")
                e("mov rdi, newline")
                e("xor rax, rax")
                e("call printf")
                e("add rsp, 32")
                e("leave")
                e("ret")

                for procedure in info.procedures:
                    proc_code, proc_data = procedure.emit(info)
                    e(proc_code)
                    d(proc_data)

                e("section '.bss' writeable")
                e("pillow_stack rb 4096")

                e("section '.data' writeable")
                e(data_section)
                e("int_fmt db \"%d\", 0")
                e("flo_fmt db \"%g\", 0")
                e("str_fmt db \"%s\", 0")
                e("space db 32, 0")
                e("newline db 10, 0")

                e("section '.note.GNU-stack'")

            case Target.WINDOWS:
                e("xor rcx, rcx")
                e("call [ExitProcess]")

                e("print_int:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 0x20")
                e("mov rdx, rax")
                e("lea rcx, [int_fmt]")
                e("call [printf]")
                e("leave")
                e("ret")

                e("print_flo:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 0x20")
                e("movq rdx, xmm0")
                e("lea rcx, [flo_fmt]")
                e("call [printf]")
                e("leave")
                e("ret")

                e("print_str:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 0x20")
                e("mov rdx, rax")
                e("lea rcx, [str_fmt]")
                e("call [printf]")
                e("leave")
                e("ret")

                e("print_spc:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 0x20")
                e("lea rcx, [space]")
                e("call [printf]")
                e("leave")
                e("ret")

                e("print_ln:")
                e("push rbp")
                e("mov rbp, rsp")
                e("sub rsp, 0x20")
                e("lea rcx, [newline]")
                e("call [printf]")
                e("leave")
                e("ret")


                for procedure in info.procedures:
                    proc_code, proc_data = procedure.emit(info)
                    e(proc_code)
                    d(proc_data)

                e("section '.bss' readable writeable")
                e("pillow_stack rb 4096")

                e("section '.data' data readable writeable")
                e(data_section)
                e("int_fmt db \"%d\", 0")
                e("flo_fmt db \"%g\", 0")
                e("str_fmt db \"%s\", 0")
                e("space db 32, 0")
                e("newline db 10, 0")

                e("section '.idata' import data readable writeable")
                e("library kernel32, 'kernel32.dll', msvcrt, 'msvcrt.dll'")
                e("import kernel32, ExitProcess, 'ExitProcess'")
                e("import msvcrt, printf, 'printf', malloc, 'malloc'")

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
        assembly, _ = emit(list(enumerate(lex(f.read()))), EmitInfo(target, True, procedures=[]))
        compile(assembly, target, output_file)


if __name__ == "__main__":
    main()
