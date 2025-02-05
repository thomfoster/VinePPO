import os
import shutil
import urllib.request
import zipfile
import glob

def main():
    # 1. Download the zip file
    zip_url = "https://github.com/McGill-NLP/VinePPO/releases/download/latest/wandb_export_root.zip"
    zip_name = "wandb_export_root.zip"
    urllib.request.urlretrieve(zip_url, zip_name)

    # 2. Create the data directory
    os.makedirs("data", exist_ok=True)

    # 3. Extract the downloaded zip
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall("zip_root")

    # 4. Process inner zip files
    datasets_path = os.path.join("zip_root", "wandb_export_root", "datasets")
    for archive in glob.glob(os.path.join(datasets_path, "*.zip")):
        base_name = os.path.basename(archive)
        ds_name = base_name.replace("data-", "").replace(".zip", "")
        target_dir = os.path.join("data", ds_name)
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        # Copy unzipped content to data
        for item in os.listdir(target_dir):
            shutil.move(os.path.join(target_dir, item), "data")
        shutil.rmtree(target_dir, ignore_errors=True)

    # 5. Remove zip_root and the downloaded zip
    shutil.rmtree("zip_root", ignore_errors=True)
    os.remove(zip_name)

if __name__ == "__main__":
    main()