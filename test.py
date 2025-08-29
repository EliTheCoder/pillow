#!/usr/bin/env python3

import subprocess
import os

result_a = subprocess.run(["python3", "pillow.py", "test.pilo"])
assert result_a.returncode == 0
result_b = subprocess.run(["test.exe" if os.name == "nt" else "./test"], capture_output = True, text = True)

expected_output = """4
0
20
80 100
123456
5
2
2
33
5
2
11
9
444
444
0
1
0
0
7
0
2
3
6
4
3
2
1
0
9
62
65
strimg cheese
2
4
3
3 1 5 4 2
7 10 9 8 7
"""

if result_b.stdout != expected_output:
    print(f"Expected this output of length {len(expected_output)}:")
    print(expected_output)
    print(f"Found this output of length {len(result_b.stdout)}:")
    print(result_b.stdout)
else:
    print("Test passed")

