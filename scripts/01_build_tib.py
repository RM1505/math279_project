import pandas as pd
import py7zr
import re
from pathlib import Path
import tempfile
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from tib import compute_tib

_NAME_RE = re.compile(r"^(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_")


DATA_RAW = Path("data/raw")
DATA_OUT = Path("data/processed/tib")


def process_csv_file(args):
    csv_file, out_dir_base = args
    
    m = _NAME_RE.match(csv_file.name)
    if m is None:
        return None

    ticker = m.group("ticker")
    date = m.group("date")

    out_dir = out_dir_base / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{date}.csv"
    if out_path.exists():
        return None

    df = pd.read_csv(csv_file, usecols=["time", " bid_size_1", "ask_size_1"])
    tib_df = compute_tib(df)
    
    tib_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    idx = pd.read_csv("data/index.csv")

    for r in tqdm(idx.itertuples(index=False), total=len(idx), desc="Archives"):
        archive_path = DATA_RAW / r.archive

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                csv_names = [n for n in z.getnames() if n.endswith(".csv")]
                z.extract(path=tmpdir, targets=csv_names)
            
            csv_files = list(tmpdir_path.rglob("*.csv"))
            
            n_processes = max(1, cpu_count() - 1)
            with Pool(processes=n_processes) as pool: # hopefully will run faster multi-threaded :shrug:
                args_list = [(f, DATA_OUT) for f in csv_files]
                list(tqdm(pool.imap_unordered(process_csv_file, args_list), 
                         total=len(csv_files), 
                         desc=f"  {r.archive}", 
                         leave=False))