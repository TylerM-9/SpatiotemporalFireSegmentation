import os
import zipfile
from notion2md.exporter.block import MarkdownExporter

# Define Notion Page ID and output paths
NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID")
EXTRACT_FOLDER = "notion_export"

# Export Notion page as a Markdown ZIP file
MarkdownExporter(block_id=NOTION_PAGE_ID, output_path='.', download=True).export()

# Unzip the file
with zipfile.ZipFile(NOTION_PAGE_ID + ".zip", "r") as zip_ref:
    zip_ref.extractall(EXTRACT_FOLDER)

# Find the exported Markdown file
for root, _, files in os.walk(EXTRACT_FOLDER):
    for file in files:
        if file.endswith(".md"):  # Find the first .md file
            old_md_path = os.path.join(root, file)
            new_md_path = os.path.join(root, "README.md")
            os.rename(old_md_path, new_md_path)
            print(f"Renamed {file} to README.md")
            break  # Stop after renaming the first markdown file

# Cleanup: Remove extracted folder and ZIP file
os.remove(NOTION_PAGE_ID + ".zip")
