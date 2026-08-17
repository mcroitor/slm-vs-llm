"""
Script to prepare a knowledge base for RAG from a folder of Markdown files.
It reads all .md files from the 'data' directory and saves them as a JSON list.
"""

import os
import json
import argparse
from typing import List

def load_markdown_files(data_dir: str) -> List[str]:
    """
    Reads all .md files in the specified directory and extracts their content.
    """
    documents = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' does not exist.")
        return documents

    print(f"Loading documents from {data_dir}...")
    
    # Walk through the directory to find all .md files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            documents.append(content)
                            print(f"Loaded: {file}")
                except Exception as e:
                    print(f"Failed to read {file}: {e}")

    print(f"Successfully loaded {len(documents)} documents.")
    return documents

def main():
    parser = argparse.ArgumentParser(description="Prepare RAG knowledge base from Markdown files")
    parser.add_argument("--data", default="data", help="Path to the folder containing .md files")
    parser.add_argument("--out", default="kb.json", help="Output JSON file path")
    args = parser.parse_args()

    # 1. Load docs
    kb_content = load_markdown_files(args.data)

    if not kb_content:
        print("No documents found. Exiting.")
        return

    # 2. Save to JSON
    try:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(kb_content, f, ensure_ascii=False, indent=2)
        print(f"\nKnowledge base saved to {args.out}")
        print(f"Total documents in KB: {len(kb_content)}")
    except Exception as e:
        print(f"Error saving knowledge base: {e}")

if __name__ == "__main__":
    main()
