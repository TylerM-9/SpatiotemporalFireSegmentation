import os
import requests

NOTION_API_KEY = os.getenv("NOTION_KEY")
NOTION_PAGE_ID = "1afad623120880b8b6bce3f9e8e3d59d"
NOTION_URL = f"https://api.notion.com/v1/blocks/{NOTION_PAGE_ID}/children"

HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

def fetch_notion_content():
    response = requests.get(NOTION_URL, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        content = []
        for block in data["results"]:
            if block["type"] == "paragraph":
                text = block["paragraph"]["rich_text"]
                content.append(" ".join([t["text"]["content"] for t in text]))
        return "\n".join(content)
    else:
        print("Failed to fetch Notion content:", response.text)
        return ""

if __name__ == "__main__":
    notion_content = fetch_notion_content()
    if notion_content:
        with open("README.md", "w", encoding="utf-8") as file:
            file.write("# Updated from Notion\n\n")
            file.write(notion_content)
