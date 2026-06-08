import os
import json
from pathlib import Path
from markitdown import MarkItDown

def main():
    landing_dir = Path("data/landing")
    standardized_dir = Path("data/standardized")
    
    md = MarkItDown()
    
    # Walk through the landing directory
    for root, _, files in os.walk(landing_dir):
        for file in files:
            if file == ".gitkeep":
                continue
                
            file_path = Path(root) / file
            
            # Determine the relative path to maintain directory structure
            rel_path = file_path.relative_to(landing_dir)
            out_dir = standardized_dir / rel_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            
            out_file_name = f"{file_path.stem}.md"
            out_file_path = out_dir / out_file_name
            
            print(f"Converting {file_path}...")
            try:
                # If it's a JSON file from Task 2, we can just extract the markdown content
                # But to strictly follow "use MarkItDown", we can check if MarkItDown supports JSON.
                # MarkItDown typically supports JSON, but it might output a code block.
                # Let's try to extract 'content' if it's our news JSON, otherwise use MarkItDown.
                if file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if "content" in data and "metadata" in data:
                        # Write the markdown directly since it's already extracted by crawl4ai
                        markdown_content = f"# {data['metadata'].get('title', 'Unknown Title')}\n\n"
                        markdown_content += f"**Source:** {data['metadata'].get('source_url', '')}\n"
                        markdown_content += f"**Crawled at:** {data['metadata'].get('crawled_at', '')}\n\n"
                        markdown_content += data["content"]
                        
                        with open(out_file_path, 'w', encoding='utf-8') as f:
                            f.write(markdown_content)
                        print(f"  -> Extracted JSON content to {out_file_path}")
                        continue
                        
                # Default case: use MarkItDown (for PDF, DOCX, HTML, etc.)
                result = md.convert(str(file_path))
                with open(out_file_path, 'w', encoding='utf-8') as f:
                    f.write(result.text_content)
                print(f"  -> Converted to {out_file_path}")
            except Exception as e:
                print(f"  -> Error converting {file_path}: {e}")

if __name__ == "__main__":
    main()
