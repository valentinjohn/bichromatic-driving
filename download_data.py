import requests
import zipfile
import io
import os
from pathlib import Path

# === Settings ===
zip_url = "https://data.4tu.nl/file/bb43fe1d-f503-49e8-9f17-ce7d734f015d/6ade02b4-3aaa-4e6a-bb92-232b4580d100"
target_subfolder = "bichromatic-driving/data/"                         # ← the folder inside the ZIP to extract
output_folder = "data"                   # ← destination folder

# === Resolve paths ===
script_dir = Path(__file__).parent.resolve()
extract_dir = script_dir / output_folder
extract_dir.mkdir(parents=True, exist_ok=True)

# check whether data has already been downloaded
if extract_dir.exists():
    # check whether the folder is empty
    if any(extract_dir.iterdir()):
        print(f"Some data already exists in '{extract_dir}'. Skip downloading.")
        exit(0)

# === Download the ZIP file into memory ===
print(f"Downloading data from {zip_url} ...")
response = requests.get(zip_url)
# response.raise_for_status()

# === Extract target subfolder exactly as-is ===
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    for member in z.namelist():
        # Only extract files that are inside the target subfolder
        if member.startswith(target_subfolder) and not member.endswith("/"):
            # Keep the full path after the subfolder
            relative_path = Path(member).relative_to(target_subfolder)
            destination_path = extract_dir / relative_path

            # Create folders if needed
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with z.open(member) as source_file, open(destination_path, "wb") as target_file:
                target_file.write(source_file.read())

print(f"✅ Extracted data into '{extract_dir}'.")



