'''
Question1 - Computibility and Complexity
Hongze Lin
'''
# ==================== Q1(b) ====================
import random

def build_tm_transitions():  
# Build and return the full TM transition table that encodes the 1^n 0 1^m -> 1^(n+m) behavior.
    """
    Transition function of the Turing machine.
    Tape alphabet: {'0', '1'}, where '0' is also used as blank.
    Actions: L, R, W1, W0, T (halt).
    """
    delta = {}

# A: Scan right in the left block 1^n until the delimiter 0.
    delta[("q0", "1")] = ("R", "q0")
    delta[("q0", "0")] = ("R", "qInspect")

# B: Move the delimiter 0 rightward by repeatedly applying 01 -> 10.
    delta[("qInspect", "1")] = ("W0", "qWL")
    delta[("qWL", "0")] = ("L", "qW1")
    delta[("qW1", "0")] = ("W1", "qBack")
    delta[("qBack", "1")] = ("R", "qDelim")
    delta[("qDelim", "0")] = ("R", "qInspect")

# Stop bubbling when no 1 exists to the right.
    delta[("qInspect", "0")] = ("L", "qReturn")

# C: Return to the leftmost 1 and halt.
    delta[("qReturn", "0")] = ("L", "qGoLeft")
    delta[("qGoLeft", "1")] = ("L", "qGoLeft")
    delta[("qGoLeft", "0")] = ("R", "qHalt")
    delta[("qHalt", "1")] = ("T", "qHalt")

    return delta

def ensure_tape_cell(tape, head):  
# Guarantee head points to a valid cell by extending tape left/right with blank symbol '0' when out of bounds.
    """
    Ensure tape[head] is valid by extending the tape with '0' when needed.
    """
    while head < 0:
        tape.insert(0, "0")
        head += 1
    while head >= len(tape):
        tape.append("0")
    return tape, head

def run_tm(input_str, transitions=None, max_steps=1_000_000, verbose=False):  
# Execute the TM step-by-step on input string and return output plus final machine state details.
    """
    Simulate input 1^n 0 1^m and output merged 1^{n+m}.
    Return (output_ones, final_tape, final_head, steps_used).
    """
    if transitions is None:
        transitions = build_tm_transitions()

    if not input_str or input_str[0] != "1":
        raise ValueError("Input tape must start with '1'.")

    # Add one trailing blank '0' to simplify right-side checks.
    tape = list(input_str) + ["0"]
    head = 0
    state = "q0"
    steps = 0

    while steps < max_steps:
        tape, head = ensure_tape_cell(tape, head)
        sym = tape[head]

        transition = transitions.get((state, sym))
        if transition is None:
            raise RuntimeError(f"No transition for state={state}, sym={sym}, head={head}.")
        action, next_state = transition

        if verbose:
            left = max(0, head - 10)
            right = min(len(tape), head + 11)
            print(f"step={steps:6d} state={state:8s} read={sym} action={action:2s} next={next_state:8s}")
            print("".join(tape[left:right]))
            print(" " * (head - left) + "^")

        if action == "L":
            head -= 1
        elif action == "R":
            head += 1
        elif action == "W1":
            tape[head] = "1"
        elif action == "W0":
            tape[head] = "0"
        elif action == "T":
            break
        else:
            raise ValueError(f"Action Error")

        state = next_state
        steps += 1
    else:
        raise RuntimeError("Steps Error")

    if "1" not in tape:
        return "", tape, head, steps
    first_one = tape.index("1")

    j = first_one
    while j < len(tape) and tape[j] == "1":
        j += 1
    output = "".join(tape[first_one:j])
    return output, tape, head, steps

# Random Test for Turing Coding
rng = random.Random(42)
num_tests = 20
max_m = 20
max_n = 20
failed_tests = 0
for _ in range(num_tests):
    n = rng.randint(1, max_n)  # Enforce leftmost symbol is '1'.
    m = rng.randint(0, max_m)
    input_str = "1" * n + "0" + "1" * m
    if input_str[0] != "1":
        failed_tests += 1
        print(f"Test failed: invalid random tape {input_str}")
        continue
    output, final_tape, final_head, steps = run_tm(input_str)
    expected_output = "1" * (n + m)
    if output != expected_output:
        failed_tests += 1
        print(
            f"Test failed: input={input_str}, expected={expected_output}, got={output}, "
            f"head={final_head}, steps={steps}"
        )
if failed_tests == 0:
    print(f"All {num_tests} random tests passed.")
else:
    print(f"{failed_tests}/{num_tests} random tests failed.")


# ==================== Q1(d) ====================
def binarysearch(list, target):
    # Iterative binary search on a sorted list.
    """
    Return the index of target in list if found, otherwise return 'fail'.
    list should be sorted in non-decreasing order.
    """
    left = 0
    right = len(list) - 1

    while left <= right:
        middle = (left + right) // 2
        value = list[middle]

        if value == target:
            return middle
        if target < value:
            right = middle - 1
        else:
            left = middle + 1

    return "Q1(d):fail"

# Random Test for Q1(d)
def run_tests_Q1_d():
    rng = random.Random(42) 
    num_tests = 20
    max_len = 30
    value_low = -20
    value_high = 20
    failed = 0
    for i in range(1, num_tests + 1):
        size = rng.randint(0, max_len)
        list = sorted(rng.randint(value_low, value_high) for _ in range(size))
        # Pick an existing value half of the time, otherwise use a random value.
        if size > 0 and rng.random() < 0.5:
            target = list[rng.randint(0, size - 1)]
        else:
            target = rng.randint(value_low, value_high)
        out = binarysearch(list, target)
        expected_indices = [idx for idx, value in enumerate(list) if value == target]
        if len(expected_indices) == 0:
            if out != "Q1(d):fail":
                failed += 1
                print(f"Test {i} failed: list={list}, target={target}, expected=fail, got={out}")
        else:
            if out == "Q1(d):fail" or out not in expected_indices:
                failed += 1
                print(
                    f"Test {i} failed: list={list}, target={target}, "
                    f"expected one of {expected_indices}, got={out}"
                )
    if failed == 0:
        print(f"Q1(d): all {num_tests} random tests passed.")
    else:
        print(f"Q1(d): {failed}/{num_tests} random tests failed.")
run_tests_Q1_d()


# ==================== Q1(f) ====================
import random
import math
import numpy as np

def binarysearch_count(list, target):
    """
    Iterative binary search that counts loop iterations.
    Return (index_or_fail, iterations).
    """
    left = 0
    right = len(list) - 1
    iters = 0

    while left <= right:
        iters += 1
        middle = (left + right) // 2
        value = list[middle]

        if value == target:
            return middle, iters
        if target < value:
            right = middle - 1
        else:
            left = middle + 1

    return "fail", iters

def run_tests_Q1_f(num_tests=300, seed=42):
    """
    Random tests for Q1(f): validate correctness and iteration bounds.
    """
    rng = random.Random(seed)
    failed = 0
    randint = rng.randint
    randrange = rng.randrange

    for i in range(1, num_tests + 1):
        n = randint(1, 300)
        list = sorted(randint(-1000, 1000) for _ in range(n))

        # Hit case only: target is always selected from list.
        target = list[randrange(n)]

        out, iters = binarysearch_count(list, target)

        if out == "fail" or list[out] != target:
            failed += 1
            print(f"Test {i} failed (hit): target={target}, got={out}")

        max_iters = math.ceil(math.log2(n + 1)) + 1
        if iters < 1 or iters > max_iters:
            failed += 1
            print(
                f"Test {i} failed (iters): n={n}, iters={iters}, "
                f"expected between 1 and {max_iters}"
            )

    if failed == 0:
        print(f"Q1(f): all {num_tests} random tests passed.")
    else:
        print(f"Q1(f): {failed}/{num_tests} random tests failed.")

def simulate_avg_iterations(lengths, trials_per_n=1000, seed=42):
    """
    For each n in lengths, sample random target positions and measure
    average successful-search iterations. Using list = [0, 1, ..., n-1]
    avoids repeated sorting.
    """
    rng = random.Random(seed)
    ns = []
    avg_iters = []
    randrange = rng.randrange

    for n in lengths:
        list = [i for i in range(n)]
        total = sum(binarysearch_count(list, randrange(n))[1] for _ in range(trials_per_n))

        ns.append(n)
        avg_iters.append(total / trials_per_n)

    return ns, avg_iters

def fit_log_model_with_polyfit(ns, avg_iters):
    """
    Fit avg_iters ≈ a * log2(n) + b with numpy.polyfit and return R^2.
    """
    x = np.log2(np.asarray(ns, dtype=float))
    y = np.asarray(avg_iters, dtype=float)
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return a, b, r2

def report_q1_f_results(ns, avg_iters):
    print("\nData (n, avg_iterations):")
    for n, av in zip(ns, avg_iters):
        print(f"n={n:6d}  avg_iters={av:.4f}")

    print("\navg_iters / log2(n):")
    log2_ns = [math.log2(n) for n in ns]
    for n, av, log2_n in zip(ns, avg_iters, log2_ns):
        ratio = av / log2_n
        print(f"n={n:6d}  ratio={ratio:.4f}")

    a, b, r2 = fit_log_model_with_polyfit(ns, avg_iters)
    print("\nPolyfit evaluation for O(log2 n) hypothesis:")
    print("Model: avg_iters ≈ a * log2(n) + b")
    print(f"a = {a:.6f}")
    print(f"b = {b:.6f}")
    print(f"R^2 = {r2:.6f}")

# Run Q1(f) tests and simulation.
def run_q1_f(num_tests=20, test_seed=42, trials_per_n=1000, sim_seed=42):
    # Run tests and the average-iteration experiment.
    run_tests_Q1_f(num_tests=num_tests, seed=test_seed)
    lengths = [2**k for k in range(3, 16)]  # 8, 16, 32, ..., 32768
    ns, avg_iters = simulate_avg_iterations(lengths, trials_per_n=trials_per_n, seed=sim_seed)
    report_q1_f_results(ns, avg_iters)
run_q1_f()