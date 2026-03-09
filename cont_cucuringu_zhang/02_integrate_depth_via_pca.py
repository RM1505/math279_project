from pathlib import Path
from integrated_ofi import calculate_integrated_ofi
from tqdm import tqdm

base = Path("data/processed/minutely_deep")

folders = [f for f in base.iterdir() if f.is_dir()]

for folder in tqdm(folders, desc="Processing tickers"):
    calculate_integrated_ofi(folder.name)