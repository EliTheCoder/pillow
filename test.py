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
1
0
1
0
4 5 6 7.2 beans -4.9 -6 
2
{x=9.2 y=7.1}
{x=7.1 y=9.2}
"""

if result_b.stdout != expected_output:
    print(f"Expected this output of length {len(expected_output)}:")
    print(expected_output)
    print(f"Found this output of length {len(result_b.stdout)}:")
    print(result_b.stdout)
    with open("test_e.txt", "w") as f:
        f.write(expected_output)
    with open("test_f.txt", "w") as f:
        f.write(result_b.stdout)
else:
    print("Test passed")

