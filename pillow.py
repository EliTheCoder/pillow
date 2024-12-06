#!/usr/bin/env python3

from enum import Enum, auto
from typing import Any
from subprocess import run, DEVNULL
from sys import argv
from pathlib import Path
import os

class TokenType(Enum):
    INTEGER = auto()
    STRING = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    INC = auto()
    DEC = auto()
    DUP = auto()
    SWP = auto()
    NOT = auto()
    POP = auto()
    PRINT = auto()
    PRINTLN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    END = auto()


class Token():
    def __init__(self, token_type: TokenType, value: Any = None):
        self.token_type = token_type
        self.value = value

    def __repr__(self) -> str:
        return self.token_type.name.lower() + (f" {self.value}" if self.value is not None else "")
        
class PillowType(Enum):
    INT = auto()
    STR = auto()
    
    def __repr__(self) -> str:
        return self.name.lower()

block_stack: list[tuple[int, Token]] = []
def lex_token(tok: str, i: int) -> Token:
    if tok.isnumeric(): return Token(TokenType.INTEGER, int(tok))
    if tok.startswith("\"") and tok.endswith("\""): return Token(TokenType.STRING, tok[1:-1])
    tok = tok.lower()
    if tok == "+": return Token(TokenType.ADD)
    if tok == "-": return Token(TokenType.SUB)
    if tok == "*": return Token(TokenType.MUL)
    if tok == "/": return Token(TokenType.DIV)
    if tok == "%": return Token(TokenType.MOD)
    if tok == "++": return Token(TokenType.INC)
    if tok == "--": return Token(TokenType.DEC)
    if tok == "dup": return Token(TokenType.DUP)
    if tok == "swp": return Token(TokenType.SWP)
    if tok == "pop": return Token(TokenType.POP)
    if tok == "!": return Token(TokenType.NOT)
    if tok == "print": return Token(TokenType.PRINT)
    if tok == "println": return Token(TokenType.PRINTLN)
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
    if tok == "end":
        assert len(block_stack) > 0, "Mismatched end"
        opening_i, opening_tok = block_stack.pop()
        opening_tok.value = i
        return Token(TokenType.END, opening_i)
    raise ValueError(f"Unexpected token {tok}")

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
        else:
            start = i
            while i < len(code) and not code[i].isspace():
                i += 1
            tokens.append(lex_token(code[start:i], len(tokens)))
    assert len(block_stack) == 0, f"Mismatched {block_stack[-1]}"
    return tokens

class Target(Enum):
    LINUX = auto()
    WINDOWS = auto()

def emit(code: list[Token], target: Target) -> str:
    output = ""
    data_section = ""

    def e(x: str) -> None:
        nonlocal output
        output += x + "\n"

    def d(x: str) -> None:
        nonlocal data_section
        data_section += x + "\n"

    type_stack: list[PillowType] = []

    def type_stack_op(tok: Token, takes: list[PillowType], gives: list[PillowType]) -> None:
        nonlocal type_stack
        if len(takes) > 0:
            assert type_stack[-len(takes):] == takes, f"Instruction {tok} takes {takes} but found {type_stack[-len(takes):]}"
            type_stack = type_stack[:-len(takes)]
        type_stack += gives

    match target:
        case Target.LINUX:
            e("format ELF64")

            e("public main")
            e("extrn exit")
            e("extrn printf")
            e("section '.text' executable")

        case Target.WINDOWS:
            e("format PE64")
            e("entry main")

            e("include 'win64a.inc'")
            e("section '.text' code executable")

    e("main:")

    block_type_stack: list[list[PillowType]] = []

    for i, tok in enumerate(code):

        def t(takes: list[PillowType], gives: list[PillowType]) -> None:
            type_stack_op(tok, takes, gives)
        
        match tok.token_type:
            case TokenType.INTEGER:
                t([], [PillowType.INT])
                e("push " + str(tok.value))
            case TokenType.STRING:
                t([], [PillowType.STR])
                d("string_" + str(i) + " db \"" + tok.value + "\", 0")
                e("push string_" + str(i))
            case TokenType.ADD:
                t([PillowType.INT, PillowType.INT], [PillowType.INT])
                e("pop rbx")
                e("pop rax")
                e("add rax, rbx")
                e("push rax")
            case TokenType.SUB:
                t([PillowType.INT, PillowType.INT], [PillowType.INT])
                e("pop rbx")
                e("pop rax")
                e("sub rax, rbx")
                e("push rax")
            case TokenType.MUL:
                t([PillowType.INT, PillowType.INT], [PillowType.INT])
                e("pop rbx")
                e("pop rax")
                e("imul rbx")
                e("push rax")
            case TokenType.DIV:
                t([PillowType.INT, PillowType.INT], [PillowType.INT])
                e("pop rbx")
                e("pop rax")
                e("cqo")
                e("idiv rbx")
                e("push rax")
            case TokenType.MOD:
                t([PillowType.INT, PillowType.INT], [PillowType.INT])
                e("pop rbx")
                e("pop rax")
                e("cqo")
                e("idiv rbx")
                e("push rdx")
            case TokenType.INC:
                t([PillowType.INT], [PillowType.INT])
                e("pop rax")
                e("inc rax")
                e("push rax")
            case TokenType.DEC:
                t([PillowType.INT], [PillowType.INT])
                e("pop rax")
                e("dec rax")
                e("push rax")
            case TokenType.DUP:
                type_stack.append(type_stack[-1])
                e("mov rax, [rsp]")
                e("push rax")
            case TokenType.SWP:
                a = type_stack.pop()
                b = type_stack.pop()
                type_stack.append(a)
                type_stack.append(b)
                e("pop rax")
                e("pop rbx")
                e("push rax")
                e("push rbx")
            case TokenType.POP:
                t([PillowType.INT], [])
                e("add rsp, 8")
            case TokenType.NOT:
                t([PillowType.INT], [PillowType.INT])
                e("pop rax")
                e("test rax, rax")
                e("sete al")
                e("movzx rax, al")
                e("push rax")
            case TokenType.PRINT:
                match type_stack[-1]:
                    case PillowType.INT:
                        t([PillowType.INT], [])
                        e("pop rax")
                        e("call print_int")
                    case PillowType.STR:
                        t([PillowType.STR], [])
                        e("pop rax")
                        e("call print_str")
                    case _:
                        assert False, f"Print expected int or str, found {type_stack[-1]}"
            case TokenType.PRINTLN:
                match type_stack[-1]:
                    case PillowType.INT:
                        t([PillowType.INT], [])
                        e("pop rax")
                        e("call print_int")
                    case PillowType.STR:
                        t([PillowType.STR], [])
                        e("pop rax")
                        e("call print_str")
                    case _:
                        assert False, f"Println expected int or str, found {type_stack[-1]}"
                e("call print_ln")
            case TokenType.IF:
                t([PillowType.INT], [])
                block_type_stack.append(type_stack.copy())
                e("pop rax")
                e("test rax, rax")
                e("jz label_" + str(tok.value))
            case TokenType.ELSE:
                old_block_type_stack = block_type_stack.pop()
                block_type_stack.append(type_stack.copy())
                type_stack = old_block_type_stack
                e("jmp label_" + str(tok.value))
                e("label_" + str(i) + ":")
            case TokenType.WHILE:
                block_type_stack.append(type_stack.copy())
                e("label_" + str(i) + ":")
                e("mov rax, [rsp]")
                e("test rax, rax")
                e("jz label_" + str(tok.value))
            case TokenType.END:
                opening_token = code[tok.value]
                old_block_type_stack = block_type_stack.pop()
                match opening_token.token_type:
                    case TokenType.IF:
                        assert old_block_type_stack == type_stack, f"If statement must leave the stack the same as before\nStarted with {old_block_type_stack} and got {type_stack}"
                    case TokenType.ELSE:
                        assert old_block_type_stack == type_stack, f"If else statement must leave the stack the same in both branches\nIf branch got {old_block_type_stack}, else branch got {type_stack}"
                    case TokenType.WHILE:
                        assert old_block_type_stack == type_stack, f"While loop must leave the stack the same as before\nStarted with {old_block_type_stack} and got {type_stack}"
                        e("jmp label_" + str(tok.value))
                e("label_" + str(i) + ":")
            case _:
                raise Exception(f"Token type {tok.token_type} not implemented")
        e("")


    match target:
        case Target.LINUX:
            e("xor rax, rax")
            e("call exit")

            e("print_int:")
            e("sub rsp, 32")
            e("mov rdi, fmt")
            e("mov rsi, rax")
            e("xor rax, rax")
            e("call printf")
            e("add rsp, 32")
            e("ret")

            e("print_str:")
            e("sub rsp, 32")
            e("mov rdi, rax")
            e("xor rax, rax")
            e("call printf")
            e("add rsp, 32")
            e("ret")

            e("print_ln:")
            e("sub rsp, 32")
            e("mov rdi, newline")
            e("xor rax, rax")
            e("call printf")
            e("add rsp, 32")
            e("ret")

            e("section '.data' writeable")
            e(data_section)
            e("fmt db \"%d\", 0")
            e("newline db 10, 0")

            e("section '.note.GNU-stack'")

        case Target.WINDOWS:
            e("xor rcx, rcx")
            e("call [ExitProcess]")

            e("print_int:")
            e("sub rsp, 32")
            e("mov rcx, fmt")
            e("mov rdx, rax")
            e("call [printf]")
            e("add rsp, 32")
            e("ret")

            e("print_str:")
            e("sub rsp, 32")
            e("mov rcx, rax")
            e("call [printf]")
            e("add rsp, 32")
            e("ret")

            e("print_ln:")
            e("sub rsp, 32")
            e("mov rcx, newline")
            e("call [printf]")
            e("add rsp, 32")
            e("ret")

            e("section '.data' data readable writeable")
            e(data_section)
            e("fmt db \"%d\", 0")
            e("newline db 10, 0")

            e("section '.idata' import data readable writeable")
            e("library kernel32, 'kernel32.dll', msvcrt, 'msvcrt.dll'")
            e("import kernel32, ExitProcess, 'ExitProcess'")
            e("import msvcrt, printf, 'printf'")

    return output

def compile(assembly: str, target: Target, output_path: Path) -> None:
    with open(output_path.with_suffix(".s"), "w") as f:
        f.write(assembly)
    fasm_result = run(["fasm", output_path.with_suffix(".s")], stdout = DEVNULL)
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
        compile(emit(lex(f.read()), target), target, output_file)


if __name__ == "__main__":
    main()
