from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm

from ofi import worker

def main():
    root = Path(r"data/processed/minutely")
    ticker_dirs = [p for p in root.iterdir() if p.is_dir()]

    out_dir = root.parent / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (str(td), str(out_dir / f"{td.name}_ofi_daily.csv"), 0.01, "time", "ofi", True)
        for td in ticker_dirs
    ]

    nprocs = max(1, min(cpu_count() - 1, 8))

    with Pool(processes=nprocs, maxtasksperchild=20) as pool:
        for _ in tqdm(pool.imap_unordered(worker, jobs, chunksize=4),
                      total=len(jobs), desc="Tickers"):
            pass


if __name__ == "__main__":
    freeze_support()
    main()