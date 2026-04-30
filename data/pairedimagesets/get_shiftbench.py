# file modified from original Apache-2.0 licensed code from https://github.com/Understanding-Visual-Datasets/VisDiff
# see LICENSE and NOTICE files in the root directory for details

import json
import logging
import os
from collections import Counter
from enum import StrEnum
from time import sleep

import shutil

import click
from tqdm import tqdm

def save_urls_to_file(filepath, file_urls):
    """Write a *pure text* URL list file (one URL per line).

    This is the format expected by icrawler's UrlListCrawler.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as f:
        for url in file_urls:
            if not url:
                continue
            f.write(f"{url}\n")


def save_url_infos_to_jsonl(filepath, file_urls, page_urls=None):
    """Write JSONL with URL metadata (one JSON object per line).

    Each line has at least: {"file_url": ..., "page_url": ...}.
    Useful when we want to preserve page URLs from Bing parsing.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    assert page_urls is None or len(file_urls) == len(page_urls)
    with open(filepath, 'w') as f:
        for i, url in enumerate(file_urls):
            if not url:
                continue
            page_url = page_urls[i] if page_urls else ""
            f.write(json.dumps({"file_url": url, "page_url": page_url}) + "\n")


def load_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_url_list_for_crawler(filepath: str, file_urls: list[str]) -> None:
    save_urls_to_file(filepath, file_urls)


def downloaded_urls_counter_from_dir(images_dir: str) -> Counter:
    counter: Counter = Counter()
    if not os.path.isdir(images_dir):
        return counter
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            urls = read_urls_from_xmp_metadata(path)
        except Exception:
            continue
        file_url = urls.get("file_url")
        if file_url:
            counter[file_url] += 1
    return counter


def count_files_in_dir(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    n = 0
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            n += 1
    return n

class CrawlMode(StrEnum):
    DOWNLOAD_IMAGES_DIRECTLY_FROM_BING = "images"
    SAVE_IMAGE_URLS_FROM_BING = "urls"
    DOWNLOAD_IMAGES_FROM_IMAGE_URLS = "images_from_urls"



def read_urls_from_xmp_metadata(image_path):
    try:
        import pyexiv2  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "pyexiv2 is required for reading XMP metadata. "
            "Install it (and system deps) or avoid commands that require XMP reading."
        ) from e

    with pyexiv2.Image(image_path) as img:
        metadata = img.read_xmp()
        return json.loads(metadata['Xmp.dc.source'].strip())


@click.group()
def cli():
    pass

@cli.command()
@click.option("--crawl-mode", default="images", type=click.Choice([str(e) for e in CrawlMode]))
@click.option("--base-dir", default="webcrawl-ad", help="Base directory to save crawled data")
def crawl(crawl_mode: CrawlMode = "images", base_dir: str = "webcrawl-ad"):
    crawl_impl(crawl_mode=crawl_mode, base_dir=base_dir)


def crawl_impl(*, crawl_mode: CrawlMode, base_dir: str) -> None:
    from icrawler_extensions import LinkPrinter, SourceWriterImageDownloader, SourceBingParser
    from icrawler.builtin import BingImageCrawler, UrlListCrawler

    base_dir_abs = os.path.abspath(base_dir)
    if not os.path.exists(base_dir_abs):
        raise ValueError(f"Base dir {base_dir_abs} does not exist!")

    try:
        os.remove("crawler.log")
    except OSError:
        pass

    logging.basicConfig(filename="crawler.log", level=logging.INFO)

    for filename in tqdm(["easy.jsonl", "medium.jsonl", "hard.jsonl"]):
        data = [json.loads(line) for line in open(f"{base_dir}/{filename}") if line.strip()]
        for idx, item in enumerate(tqdm(data)):
            set1, set2 = item["set1"], item["set2"]

            crawler_params = dict(feeder_threads=1,
                parser_threads=1,
                downloader_threads=16,
                )
            if crawl_mode == CrawlMode.DOWNLOAD_IMAGES_DIRECTLY_FROM_BING:
                crawler_cls = BingImageCrawler
                crawler_params["downloader_cls"] = SourceWriterImageDownloader
                crawler_params["parser_cls"] = SourceBingParser
            elif crawl_mode == CrawlMode.SAVE_IMAGE_URLS_FROM_BING:
                crawler_cls = BingImageCrawler
                crawler_params["downloader_cls"] = LinkPrinter
                crawler_params["parser_cls"] = SourceBingParser
            elif crawl_mode == CrawlMode.DOWNLOAD_IMAGES_FROM_IMAGE_URLS:
                crawler_cls = UrlListCrawler
                crawler_params["downloader_cls"] = SourceWriterImageDownloader
            else:
                raise ValueError(f"Unknown crawl mode: {crawl_mode}")

            save_base_dir = f"{base_dir}/{filename.replace('.jsonl', '')}"

            logging.info(f"##### Processing {filename} {idx}_1 ({set1}) #####")
            google_crawler = crawler_cls(
                **dict(crawler_params, storage={
                    "root_dir": f"{save_base_dir}/{idx}_1"
                })
            )
            ACCEPT_LANGUAGES = "en-US,en;q=0.5" # adapted from icrawler defaults.py
            USER_AGENT = (
                "Mozilla/5.0 (X11; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0"
            )
            HEADERS = {
                "User-Agent": USER_AGENT,
                "Accept-Language": ACCEPT_LANGUAGES,
            }
            google_crawler.session.headers.update(HEADERS)

            def make_url_safe(keyword):
                safe_keyword = keyword.strip().replace(" ", "+")
                assert all(c not in safe_keyword for c in ['/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', '&', '=']), f"Invalid character in keyword: {safe_keyword}"
                return safe_keyword

            filters = {"type": "photo", "size": ">100x100"}

            url_file_1 = f"{save_base_dir}/{idx}_1_urls.txt"
            if crawl_mode == CrawlMode.DOWNLOAD_IMAGES_FROM_IMAGE_URLS:
                assert os.path.exists(url_file_1), f"{url_file_1} does not exist!"
                google_crawler.crawl(url_file_1)
            else:
                google_crawler.crawl(keyword=make_url_safe(set1), max_num=200, filters=filters)

            if crawl_mode == CrawlMode.SAVE_IMAGE_URLS_FROM_BING:
                file_urls = google_crawler.downloader.file_urls
                page_urls = google_crawler.downloader.page_urls
                # `.txt` is the pure-URL list for UrlListCrawler.
                save_urls_to_file(url_file_1, file_urls)
                # `.jsonl` retains both file_url and page_url for provenance/debugging.
                save_url_infos_to_jsonl(url_file_1.replace(".txt", ".jsonl"), file_urls, page_urls)

            logging.info(f"##### Processing {filename} {idx}_2 ({set2}) #####")
            google_crawler = crawler_cls(
                **dict(crawler_params, storage={
                    "root_dir": f"{save_base_dir}/{idx}_2"
                })
            )
            url_file_2 = f"{save_base_dir}/{idx}_2_urls.txt"
            if crawl_mode == CrawlMode.DOWNLOAD_IMAGES_FROM_IMAGE_URLS:
                assert os.path.exists(url_file_2), f"{url_file_2} does not exist!"
                google_crawler.crawl(url_file_2)
            else:
                google_crawler.crawl(keyword=make_url_safe(set2), max_num=200, filters=filters)

            if crawl_mode == CrawlMode.SAVE_IMAGE_URLS_FROM_BING:
                file_urls = google_crawler.downloader.file_urls
                page_urls = google_crawler.downloader.page_urls
                save_urls_to_file(url_file_2, file_urls)
                save_url_infos_to_jsonl(url_file_2.replace(".txt", ".jsonl"), file_urls, page_urls)
            if crawl_mode != CrawlMode.DOWNLOAD_IMAGES_FROM_IMAGE_URLS:
                sleep(30)  # be nice to the server
            else:
                sleep(0.1) # shorter sleep since we're just downloading from URLs, not hitting Bing


@cli.command()
@click.option(
    "--source-jsonl-dir",
    default="../AD-Diff_Bench",
    show_default=True,
    help="Directory containing easy.jsonl/medium.jsonl/hard.jsonl with *_images_url lists",
)
@click.option(
    "--download-base-dir",
    default="webcrawl-ad",
    show_default=True,
    help="Directory to store downloads + generated URL lists for crawl(images_from_urls)",
)
@click.option(
    "--benchmark-name",
    default="AD-Diff_Bench",
    show_default=True,
    help="Output dataset directory for release() (will contain images + per-split jsonl)",
)
@click.option("--n-sample", default=100, show_default=True, type=int)
def download_from_urls_and_release(
    source_jsonl_dir: str,
    download_base_dir: str,
    benchmark_name: str,
    n_sample: int,
):
    """Download images from existing URL lists, then run release + release_csv.

    Note: this does *not* mutate the source jsonl. The released dataset jsonl is
    derived from the images that were actually downloaded/processed.
    """

    source_jsonl_dir_abs = os.path.abspath(source_jsonl_dir)
    download_base_dir_abs = os.path.abspath(download_base_dir)

    os.makedirs(download_base_dir_abs, exist_ok=True)

    # Copy source jsonl into download_base_dir for crawl()
    for split in ("easy", "medium", "hard"):
        shutil.copy2(
            os.path.join(source_jsonl_dir_abs, f"{split}.jsonl"),
            os.path.join(download_base_dir_abs, f"{split}.jsonl"),
        )

    # Generate URL list files for crawl(images_from_urls)
    for split in ("easy", "medium", "hard"):
        data = load_jsonl(os.path.join(source_jsonl_dir_abs, f"{split}.jsonl"))
        split_dir = os.path.join(download_base_dir_abs, split)
        os.makedirs(split_dir, exist_ok=True)

        for idx, item in enumerate(data):
            set1_urls = list(item.get("set1_images_url", []) or [])
            set2_urls = list(item.get("set2_images_url", []) or [])
            url_file_1 = os.path.join(split_dir, f"{idx}_1_urls.txt")
            url_file_2 = os.path.join(split_dir, f"{idx}_2_urls.txt")
            write_url_list_for_crawler(url_file_1, set1_urls)
            write_url_list_for_crawler(url_file_2, set2_urls)

    # Download images via crawl(images_from_urls)
    crawl_impl(crawl_mode=CrawlMode.DOWNLOAD_IMAGES_FROM_IMAGE_URLS, base_dir=download_base_dir_abs)

    # Summarize download success without mutating source jsonl
    any_underfull: list[tuple[str, int, int, int]] = []
    for split in ("easy", "medium", "hard"):
        src_path = os.path.join(source_jsonl_dir_abs, f"{split}.jsonl")
        data = load_jsonl(src_path)

        total_expected_set1 = 0
        total_expected_set2 = 0
        total_downloaded_set1 = 0
        total_downloaded_set2 = 0
        total_files_on_disk_set1 = 0
        total_files_on_disk_set2 = 0

        for idx, item in enumerate(data):
            expected_set1_urls = [u for u in (item.get("set1_images_url", []) or []) if u]
            expected_set2_urls = [u for u in (item.get("set2_images_url", []) or []) if u]

            set1_dir = os.path.join(download_base_dir_abs, split, f"{idx}_1")
            set2_dir = os.path.join(download_base_dir_abs, split, f"{idx}_2")

            set1_downloaded = downloaded_urls_counter_from_dir(set1_dir)
            set2_downloaded = downloaded_urls_counter_from_dir(set2_dir)

            expected1 = len(expected_set1_urls)
            expected2 = len(expected_set2_urls)
            downloaded1 = sum(set1_downloaded.values())
            downloaded2 = sum(set2_downloaded.values())

            files1 = count_files_in_dir(set1_dir)
            files2 = count_files_in_dir(set2_dir)

            total_expected_set1 += expected1
            total_expected_set2 += expected2
            total_downloaded_set1 += downloaded1
            total_downloaded_set2 += downloaded2
            total_files_on_disk_set1 += files1
            total_files_on_disk_set2 += files2

            missing1 = max(0, expected1 - downloaded1)
            missing2 = max(0, expected2 - downloaded2)
            if missing1 > 0 or missing2 > 0:
                logging.warning(
                    "%s/%d missing: set1 %d/%d, set2 %d/%d",
                    split,
                    idx,
                    missing1,
                    expected1,
                    missing2,
                    expected2,
                )

            if downloaded1 < n_sample or downloaded2 < n_sample:
                any_underfull.append((split, idx, downloaded1, downloaded2))

        total_missing_set1 = max(0, total_expected_set1 - total_downloaded_set1)
        total_missing_set2 = max(0, total_expected_set2 - total_downloaded_set2)
        total_expected = total_expected_set1 + total_expected_set2
        total_downloaded = total_downloaded_set1 + total_downloaded_set2
        total_missing = total_missing_set1 + total_missing_set2
        total_files_on_disk = total_files_on_disk_set1 + total_files_on_disk_set2

        print(f"DOWNLOAD SUMMARY {split}:")
        print(
            f"  set1 expected={total_expected_set1}, downloaded={total_downloaded_set1}, "
            f"missing={total_missing_set1}, files_on_disk={total_files_on_disk_set1}"
        )
        print(
            f"  set2 expected={total_expected_set2}, downloaded={total_downloaded_set2}, "
            f"missing={total_missing_set2}, files_on_disk={total_files_on_disk_set2}"
        )
        print(
            f"  total expected={total_expected}, downloaded={total_downloaded}, "
            f"missing={total_missing}, files_on_disk={total_files_on_disk}"
        )

    # Release benchmark (processed JPGs) and CSV
    release_impl(n_sample=n_sample, base_dir=download_base_dir_abs, benchmark_name=benchmark_name)
    release_csv_impl(benchmark_name=benchmark_name)

    # Print underfull sets (<n_sample) at end
    for split, idx, n1, n2 in any_underfull:
        if n1 < n_sample or n2 < n_sample:
            print(f"FINAL SIZE {split}/{idx}: set1={n1}, set2={n2} (target={n_sample})")


def process_image_to_jpg(input_path, output_path, resolution=512):
    """
    Convert and resize the image to JPG format with max dimension of 512.

    Parameters:
    - input_path (str): Path to the source image.
    - output_path (str): Path to save the processed JPG image.

    Returns:
    None
    """

    from PIL import Image

    # Open the image
    with Image.open(input_path) as img:
        xmp = img.info['xmp']
        img = img.convert("RGB")

        # Get the aspect ratio
        aspect_ratio = img.width / img.height

        # Determine new dimensions based on aspect ratio
        if img.width > img.height:
            new_width = resolution
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = resolution
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        img_resized = img.resize((new_width, new_height), resample=Image.LANCZOS)

        # Save as JPG
        img_resized.save(output_path, "JPEG", xmp=xmp)
        return output_path

@cli.command()
@click.option("--n-sample", default=100, help="Number of images to sample from each set")
@click.option("--base-dir", default="webcrawl-ad", help="Base directory where crawled data is saved")
@click.option("--benchmark-name", default="AD-Diff_Bench", help="Directory name to save the final benchmark dataset")
def release(n_sample=100, base_dir="webcrawl-ad", benchmark_name="AD-Diff_Bench"):
    release_impl(n_sample=n_sample, base_dir=base_dir, benchmark_name=benchmark_name)


def release_impl(*, n_sample: int, base_dir: str, benchmark_name: str) -> None:

    os.makedirs(f"{benchmark_name}", exist_ok=True)

    for difficulty in tqdm(("easy", "medium", "hard")):
        data = [json.loads(line) for line in open(f"{base_dir}/{difficulty}.jsonl") if line.strip()]
        for idx, item in enumerate(tqdm(data)):
            os.makedirs(f"{benchmark_name}/{difficulty}/{idx}_1", exist_ok=True)
            os.makedirs(f"{benchmark_name}/{difficulty}/{idx}_2", exist_ok=True)

            set1_images = sorted(os.listdir(f"{base_dir}/{difficulty}/{idx}_1"))
            set2_images = sorted(os.listdir(f"{base_dir}/{difficulty}/{idx}_2"))
            if not (len(set1_images) >= n_sample and len(set2_images) >= n_sample):
                print(f"{difficulty}/{idx} has less than {n_sample} images")
                print(f"set1: {len(set1_images)}, set2: {len(set2_images)}")

            item["set1_images"] = []
            item["set2_images"] = []
            item["set1_images_url"] = []
            item["set2_images_url"] = []
            item["set1_pages_url"] = []
            item["set2_pages_url"] = []
            # copy these files to new folder
            image_idx_set1 = 0
            for image in set1_images:
                if image_idx_set1 >= n_sample:
                    break
                try:
                    processed_img_path = process_image_to_jpg(
                        f"{base_dir}/{difficulty}/{idx}_1/{image}",
                        f"{benchmark_name}/{difficulty}/{idx}_1/{image_idx_set1}.jpg",
                    )
                    item["set1_images"].append(processed_img_path)
                    urls = read_urls_from_xmp_metadata(processed_img_path)
                    item["set1_images_url"].append(urls["file_url"])
                    item["set1_pages_url"].append(urls.get("page_url", ""))
                except Exception as e:
                    print(f"Error processing image {base_dir}/{difficulty}/{idx}_1/{image}: {e}")
                else:
                    image_idx_set1 += 1

            image_idx_set2 = 0
            for image in set2_images:
                if image_idx_set2 >= n_sample:
                    break
                try:
                    processed_img_path = process_image_to_jpg(
                        f"{base_dir}/{difficulty}/{idx}_2/{image}",
                        f"{benchmark_name}/{difficulty}/{idx}_2/{image_idx_set2}.jpg",
                    )
                    item["set2_images"].append(processed_img_path)
                    urls = read_urls_from_xmp_metadata(processed_img_path)
                    item["set2_images_url"].append(urls["file_url"])
                    item["set2_pages_url"].append(urls.get("page_url", ""))
                except Exception as e:
                    print(f"Error processing image {base_dir}/{difficulty}/{idx}_2/{image}: {e}")
                else:
                    image_idx_set2 += 1

        # write jsonl
        with open(f"{benchmark_name}/{difficulty}.jsonl", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

@cli.command()
@click.option("--benchmark-name", default="AD-Diff_Bench", help="Directory name where the benchmark dataset is saved that needs to be converted to CSV")
def release_csv(benchmark_name="AD-Diff_Bench"):
    release_csv_impl(benchmark_name=benchmark_name)


def release_csv_impl(*, benchmark_name: str) -> None:
    # group_name is set1 or set2 name
    csv_header = "group_name,path,difference,difficuly"
    file_name = f"{benchmark_name}.csv"
    with open(file_name, "w") as f:
        f.write(csv_header + "\n")
        for difficulty in tqdm(("easy", "medium", "hard")):
            data = [json.loads(line) for line in open(f"{benchmark_name}/{difficulty}.jsonl") if line.strip()]
            for idx, item in enumerate(tqdm(data)):
                set1_name = item["set1"]
                set2_name = item["set2"]
                difference = item["difference"]

                # image_path is relative to benchmark, in the csv we want path relative to repo root
                for image_path in item["set1_images"]:
                    image_path_repo_abs = os.path.relpath(image_path, start=os.path.join(os.curdir, "..", ".."))
                    f.write(f'"{set1_name}","{image_path_repo_abs}","{difference}","{difficulty}"\n')
                for image_path in item["set2_images"]:
                    image_path_repo_abs = os.path.relpath(image_path, start=os.path.join(os.curdir, "..", ".."))
                    f.write(f'"{set2_name}","{image_path_repo_abs}","{difference}","{difficulty}"\n')
    print(f"Wrote CSV file to {file_name}")



if __name__ == "__main__":
    cli()
