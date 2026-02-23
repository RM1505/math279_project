from index_raw import index_7z_folder

if __name__ == "__main__":
    df = index_7z_folder("data/raw")
    df.to_csv("data/index.csv", index=True)