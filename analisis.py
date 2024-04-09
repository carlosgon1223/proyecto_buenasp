import pandas as pd
import subprocess
import matplotlib.pyplot as plt
def get_complexity_over_time(file_path):
    result = subprocess.run(["git", "log", "--pretty=format:%ad", "--date=short", "--", file_path], capture_output=True, text=True)
    commits = result.stdout.splitlines()
    complexities = []
    for commit in commits:
        result = subprocess.run(["git", "checkout", commit], capture_output=True, text=True)
        result = subprocess.run(["radon", "cc", file_path], capture_output=True, text=True)
        complexities.append(int(result.stdout.split()[0]))
    return commits, complexities

def plot_complexity_over_time(file_path):
    commits, complexities = get_complexity_over_time(file_path)
    df = pd.DataFrame({"Date": commits, "Complexity": complexities})
    plt.plot(df["Date"], df["Complexity"])
    plt.title(f"Cyclomatic Complexity over Time for {file_path}")
    plt.xlabel("Date")
    plt.ylabel("Complexity")
    plt.grid()
    plt.show()

file_paths = [
    "sklearn/linear_model/_logistic.py",
    "sklearn/neighbors/ridge.py",
    "sklearn/neighbors/quad_tree.py",
    "sklearn/decomposition/pca.py",
    "sklearn/metrics/pairwise.py"
]

for file_path in file_paths:
    plot_complexity_over_time(file_path)