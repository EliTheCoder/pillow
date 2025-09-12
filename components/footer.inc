if TARGET = 0
    pillow_exit:
    xor rax, rax
    call exit

    print_int:
    push rbp
    mov rbp, rsp
    mov rdi, int_fmt
    mov rsi, rax
    xor rax, rax
    call printf
    leave
    ret

    print_flo:
    push rbp
    mov rbp, rsp
    sub rsp, 8
    lea rdi, [flo_fmt]
    mov eax, 1
    call printf
    leave
    ret

    print_str:
    push rbp
    mov rbp, rsp
    mov rdi, str_fmt
    mov rsi, rax
    xor rax, rax
    call printf
    leave
    ret

    print_spc:
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rdi, space
    xor rax, rax
    call printf
    add rsp, 32
    leave
    ret

    print_ln:
    push rbp
    mov rbp, rsp
    sub rsp, 32
    mov rdi, newline
    xor rax, rax
    call printf
    add rsp, 32
    leave
    ret

    section '.bss' writeable
    pillow_stack rb 4096

    section '.note.GNU-stack'

    section '.data' writeable
    int_fmt db "%d", 0
    flo_fmt db "%g", 0
    str_fmt db "%s", 0
    space db 32, 0
    newline db 10, 0
else
    pillow_exit:
    xor rcx, rcx
    call [ExitProcess]

    print_int:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    mov rdx, rax
    lea rcx, [int_fmt]
    call [printf]
    leave
    ret

    print_flo:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    movq rdx, xmm0
    lea rcx, [flo_fmt]
    call [printf]
    leave
    ret

    print_str:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    mov rdx, rax
    lea rcx, [str_fmt]
    call [printf]
    leave
    ret

    print_spc:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    lea rcx, [space]
    call [printf]
    leave
    ret

    print_ln:
    push rbp
    mov rbp, rsp
    sub rsp, 0x20
    lea rcx, [newline]
    call [printf]
    leave
    ret

    section '.bss' readable writeable
    pillow_stack rb 4096

    section '.idata' import data readable writeable
    library kernel32, 'kernel32.dll', msvcrt, 'msvcrt.dll'
    import kernel32, ExitProcess, 'ExitProcess'
    import msvcrt, printf, 'printf', malloc, 'malloc'

    section '.data' data readable writeable
    int_fmt db "%d", 0
    flo_fmt db "%g", 0
    str_fmt db "%s", 0
    space db 32, 0
    newline db 10, 0
end if
