import numpy as np
import matplotlib.pyplot as plt

# Evaluation point
x0 = 1.0

# Differentiation step sizes: hn = 2 / 2^n for n = 1..21
n_vals = np.arange(1, 22)
h = 2.0 / (2.0 ** n_vals)

# 1. f(x) = sin(x^2)
def f1(x): return np.sin(x**2)
def df1(x): return 2 * x * np.cos(x**2)

# 2. f(x) = cos(sin(x))
def f2(x): return np.cos(np.sin(x))
def df2(x): return -np.sin(np.sin(x)) * np.cos(x)

# 3. f(x) = exp(sin(cos(x)))
def f3(x): return np.exp(np.sin(np.cos(x)))
def df3(x): return -np.exp(np.sin(np.cos(x))) * np.cos(np.cos(x)) * np.sin(x)

# 4. f(x) = ln(x+3)
def f4(x): return np.log(x + 3)
def df4(x): return 1.0 / (x + 3)

# 5. f(x) = (x+3)^0.5
def f5(x): return np.sqrt(x + 3)
def df5(x): return 0.5 / np.sqrt(x + 3)

funcs = [
    {"name": "sin(x^2)", "filename": "sin_x2", "f": f1, "df": df1},
    {"name": "cos(sin(x))", "filename": "cos_sin_x", "f": f2, "df": df2},
    {"name": "exp(sin(cos(x)))", "filename": "exp_sin_cos_x", "f": f3, "df": df3},
    {"name": "ln(x+3)", "filename": "ln_x_plus_3", "f": f4, "df": df4},
    {"name": "(x+3)^0.5", "filename": "sqrt_x_plus_3", "f": f5, "df": df5}
]

# Numerical Differentiation Formulas
def method_1(f, x, h):
    return (f(x + h) - f(x)) / h

def method_2(f, x, h):
    return (f(x) - f(x - h)) / h

def method_3(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def method_4(f, x, h):
    term1 = (4/3) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (1/3) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)
    return term1 - term2

def method_5(f, x, h):
    term1 = (3/2) * (f(x + h) - f(x - h)) / (2 * h)
    term2 = (3/5) * (f(x + 2*h) - f(x - 2*h)) / (4 * h)
    term3 = (1/10) * (f(x + 3*h) - f(x - 3*h)) / (6 * h)
    return term1 - term2 + term3

methods = [
    {"name": "Method 1: O(h)", "func": method_1},
    {"name": "Method 2: O(h)", "func": method_2},
    {"name": "Method 3: O(h^2)", "func": method_3},
    {"name": "Method 4: O(h^4)", "func": method_4},
    {"name": "Method 5: O(h^6)", "func": method_5}
]

# Plotting
# Plotting and saving figures
for idx, item in enumerate(funcs, start=1):
    plt.figure(figsize=(10, 6))

    true_val = item["df"](x0)

    for m in methods:
        approx_val = m["func"](item["f"], x0, h)
        error = np.abs(approx_val - true_val)
        plt.loglog(h, error, marker='o', markersize=4, label=m["name"])

    plt.title(f"Absolute Error of Numerical Differentiation\nFunction: {item['name']} at x = {x0}")
    plt.xlabel("Differentiation Step (h)")
    plt.ylabel("Absolute Error: |f'_{num}(x) - f'_{exact}(x)|")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()

    # Save each figure to disk
    output_filename = f"error_plot_func_{item['filename']}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")
