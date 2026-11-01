"""run_all.py -- reproduces every number, table and figure of the revised article (about 3 minutes)."""
import runpy, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for s in ["01_article_numbers.py", "02_figures.py"]:
    t0 = time.time(); print(f"\n{'=' * 20} {s} {'=' * 20}")
    sys.argv = [s]
    runpy.run_path(str(HERE / s), run_name="__main__")
    sys.stdout = sys.__stdout__                      # 01 redirects the output to its own log
    print(f"[{s} : {time.time() - t0:.0f} s]")
print("\nResultats dans outputs/ ; controle des chiffres : outputs/article_numbers_check.csv")
