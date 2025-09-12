if TARGET = 0
    format ELF64

    public main
    extrn exit
    extrn printf
    extrn malloc
    extrn realloc
    section '.text' executable

    macro spush value
        sub r12, 8
        mov r8, value
        mov qword [r12], r8
    end macro

    macro spop value
        mov value, qword [r12]
        add r12, 8
    end macro

    macro spushsd xmmreg
        sub r12, 8
        movq [r12], xmmreg
    end macro

    macro spopsd xmmreg
        movq xmmreg, [r12]
        add r12, 8
    end macro

    main:
    lea r12, [pillow_stack + 4096]
else
    format PE64
    entry main

    include 'win64a.inc'
    section '.text' code executable

    macro spush value
        sub r12, 8
        mov rax, value
        mov qword [r12], rax
    end macro

    macro spop value
        mov value, qword [r12]
        add r12, 8
    end macro

    macro spushsd xmmreg
        sub r12, 8
        movq [r12], xmmreg
    end macro

    macro spopsd xmmreg
        movq xmmreg, [r12]
        add r12, 8
    end macro

    main:
    lea r12, [pillow_stack + 4096]
end if
