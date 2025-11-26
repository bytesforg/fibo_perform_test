import time
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(20000)

# ============================================================
# Fibonacci Implementations
# ============================================================
def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n-1) + fib_recursive(n-2)

def fib_memo(n, memo):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]

def fib_dp(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def fib_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Fast Doubling (O(log n))
def fast_doubling(n):
    if n == 0:
        return (0, 1)
    a, b = fast_doubling(n // 2)
    c = a * (2*b - a)
    d = a*a + b*b
    if n % 2 == 0:
        return (c, d)
    else:
        return (d, c + d)

def fib_fast(n):
    return fast_doubling(n)[0]


# ============================================================
# Time-limited wrapper
# ============================================================
def time_limited(func, n, time_limit, memo=None):
    start = time.time()
    try:
        if memo is not None:
            func(n, memo)
        else:
            func(n)
        elapsed = time.time() - start
        if elapsed > time_limit:
            return None, True
        return elapsed, False
    except Exception:
        return None, True


# ============================================================
# MAIN SCRIPT — number limit + graph + excel
# ============================================================
def main():

    MAX_N = 5000       # <-- YOU CAN CHANGE THIS LIMIT ANYTIME
    TIME_LIMIT = 1.0   # seconds per algorithm per n

    results = {
        "n": [],
        "Brute Recursive": [],
        "Memoized Recursive": [],
        "Bottom-Up DP": [],
        "Optimized DP": [],
        "Fast Doubling": []
    }

    # Stop flags
    stop_brute = stop_memo = stop_dp = stop_opt = stop_fast = False
    memo_cache = {}

    print("\nStarting Fibonacci benchmark...\n")

    for n in range(1, MAX_N + 1):
        print(f"Calculating n = {n}")

        # -------- Brute Recursive --------
        if not stop_brute:
            t, timeout = time_limited(fib_recursive, n, TIME_LIMIT)
            if timeout:
                print(f"  → Brute Recursive stopped at n = {n}")
                stop_brute = True
                t = None
        else:
            t = None
        brute_t = t

        # -------- Memoized Recursive --------
        if not stop_memo:
            t, timeout = time_limited(fib_memo, n, TIME_LIMIT, memo_cache)
            if timeout:
                print(f"  → Memoized Recursive stopped at n = {n}")
                stop_memo = True
                t = None
        else:
            t = None
        memo_t = t

        # -------- Bottom-Up DP --------
        if not stop_dp:
            t, timeout = time_limited(fib_dp, n, TIME_LIMIT)
            if timeout:
                print(f"  → Bottom-Up DP stopped at n = {n}")
                stop_dp = True
                t = None
        else:
            t = None
        dp_t = t

        # -------- Optimized DP --------
        if not stop_opt:
            t, timeout = time_limited(fib_optimized, n, TIME_LIMIT)
            if timeout:
                print(f"  → Optimized DP stopped at n = {n}")
                stop_opt = True
                t = None
        else:
            t = None
        opt_t = t

        # -------- Fast Doubling --------
        if not stop_fast:
            t, timeout = time_limited(fib_fast, n, TIME_LIMIT)
            if timeout:
                print(f"  → Fast Doubling stopped at n = {n}")
                stop_fast = True
                t = None
        else:
            t = None
        fast_t = t

        # Save results row
        results["n"].append(n)
        results["Brute Recursive"].append(brute_t)
        results["Memoized Recursive"].append(memo_t)
        results["Bottom-Up DP"].append(dp_t)
        results["Optimized DP"].append(opt_t)
        results["Fast Doubling"].append(fast_t)

        # If all algorithms have stopped earlier, break early
        if stop_brute and stop_memo and stop_dp and stop_opt and stop_fast:
            print("\nAll algorithms have stopped by time limit.\n")
            break

    # ============================================================
    # SAVE EXCEL
    # ============================================================
    df = pd.DataFrame(results)
    df.to_excel("fib_all.xlsx", index=False, engine="openpyxl")
    print("\nSaved Excel file: fib_all.xlsx")

    # ============================================================
    # PLOT GRAPH
    # ============================================================
    cumulative = {}

    # Plot each column ignoring None
    for col in ["Brute Recursive", "Memoized Recursive", "Bottom-Up DP",
                "Optimized DP", "Fast Doubling"]:
        cumulative[col] = pd.Series(df[col]).fillna(0).cumsum()
    plt.figure(figsize=(12, 7))
    for col in ["Brute Recursive", "Memoized Recursive", "Bottom-Up DP",
                "Optimized DP", "Fast Doubling"]:
        plt.plot(df["n"], cumulative[col], label=col + " (Cumulative)")


    plt.yscale("log")     # big performance differences → log scale
    plt.xlabel("n")
    plt.ylabel("Time (seconds, log scale)")
    plt.title("Fibonacci Algorithm Performance Comparison")
    plt.legend()
    plt.grid(True)

    plt.savefig("fib_graph.png", dpi=200)
    plt.show()

    print("Saved plot: fib_graph.png")


if __name__ == "__main__":
    main()
