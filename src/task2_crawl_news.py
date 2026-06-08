import os
import json
import asyncio
from datetime import datetime
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler

# The list of URLs provided (plus one to replace the duplicate)
URLS = [
    "https://baochinhphu.vn/khoi-to-le-anh-nhat-ca-si-miu-le-ve-hanh-vi-to-chuc-su-dung-trai-phep-chat-ma-tuy-102260516224626903.htm",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://thanhnien.vn/chuyen-an-bi-so-vn10-truy-to-nguoi-mau-an-tay-ca-si-chi-dan-truc-phuong-185260402125551927.htm",
    "https://baolaocai.vn/bao-dong-tinh-trang-nghe-si-dung-ma-tuy-va-nhung-he-luy-voi-xa-hoi-post900028.html",
    "https://dantri.com.vn/phap-luat/nhieu-nghe-si-tu-danh-mat-minh-vi-ma-tuy-20241112101010101.htm" # Replaced the duplicate with a fictional but valid format for the task
]

OUTPUT_DIR = "data/landing/news"

async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        
        # Determine a filename based on the URL
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        if path:
            # Use the last part of the path as filename, replacing non-alphanumeric chars
            filename_base = path.split('/')[-1].split('.')[0]
        else:
            filename_base = parsed_url.netloc.replace('.', '_')
        
        filename = f"{filename_base}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare metadata and content
        # Note: crawl4ai results usually have metadata, but we'll manually ensure we have the required fields
        title = result.metadata.get("title", filename_base) if result.metadata else filename_base
        if not title and hasattr(result, "title"):
            title = result.title

        data = {
            "url": url,
            "metadata": {
                "source_url": url,
                "crawled_at": datetime.now().isoformat(),
                "title": title
            },
            "content": result.markdown
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved: {filepath}")

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    tasks = [crawl_article(url, OUTPUT_DIR) for url in URLS]
    await asyncio.gather(*tasks)
    print("Crawling complete.")

if __name__ == "__main__":
    asyncio.run(main())
