# code adapted from icrawler (https://github.com/hellock/icrawler) under MIT License:

# The MIT License (MIT)

# Copyright (c) 2016 Kai Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os

from icrawler.builtin import ImageDownloader, BingParser, Parser
try:
    import pyexiv2  # type: ignore
except ModuleNotFoundError:  # optional unless writing XMP metadata
    pyexiv2 = None
from bs4 import BeautifulSoup
import requests
import re
import html
import threading
import queue
import json

# adapted from https://github.com/hellock/icrawler/issues/73#issuecomment-1834390753
class LinkPrinter(ImageDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_urls = []
        self.page_urls = []

    def get_filename(self, task, default_ext):
        file_idx = self.fetched_num + self.file_idx_offset
        return '{:04d}.{}'.format(file_idx, default_ext)

    def download(self, task, default_ext, timeout=5, max_retry=3, overwrite=False, **kwargs):
        """Don't download the image and save it's URL to file_urls

        Args:
            task (dict): The task dict got from ``task_queue``.
            timeout (int): Timeout of making requests for downloading images.
            max_retry (int): the max retry times if the request fails.
            **kwargs: reserved arguments for overriding.
        """
        file_url = task["file_url"]
        page_url = task.get("page_url", None)
        task["success"] = False
        task["filename"] = None
        retry = max_retry

        if not overwrite:
            with self.lock:
                self.fetched_num += 1
                filename = self.get_filename(task, default_ext)
                if self.storage.exists(filename):
                    self.logger.info("skip downloading file %s", filename)
                    return
                self.fetched_num -= 1

        while retry > 0 and not self.signal.get("reach_max_num"):
            try:
                response = self.session.get(file_url, timeout=timeout)
            except requests.RequestException as e:
                self.logger.error(
                    "Exception caught when downloading file %s, " "error: %s, remaining retry times: %d",
                    file_url,
                    e,
                    retry - 1,
                )
            else:
                if self.reach_max_num():
                    self.signal.set(reach_max_num=True)
                    break
                elif response.status_code != 200:
                    self.logger.error("Response status code %d, file %s", response.status_code, file_url)
                    break
                elif not self.keep_file(task, response, **kwargs):
                    break
                with self.lock:
                    self.fetched_num += 1
                    filename = self.get_filename(task, default_ext)
                self.logger.info("image #%s\t%s", self.fetched_num, file_url)
                # self.storage.write(filename, response.content)  # COMMENTED OUT
                task["success"] = True
                task["filename"] = filename
                self.file_urls.append(file_url)  # ADDED
                self.page_urls.append(page_url)  # ADDED
                break
            finally:
                retry -= 1

class SourceWriterImageDownloader(ImageDownloader):
    """ICrawler pipeline component for writing source URL to XMP metadata"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_meta(self, task):
        if task["success"]:
            file_path = os.path.join(self.storage.root_dir, task["filename"])
            try:
                self.write_url_to_xmp_metadata(file_path, task["file_url"], task.get("page_url", None))
            except Exception as e:
                self.logger.error(f"Failed to write URL to XMP metadata for image from url {task['file_url']}")
                self.logger.error("Exception details:", exc_info=True)
                self.logger.error("Deleting the downloaded image.")
                try:
                    os.remove(file_path)
                except Exception as delete_error:
                    self.logger.error(f"Failed to delete image at {file_path}", exc_info=True)
                task["success"] = False
                task["filename"] = None
                print(f"Deleted corrupted image at {file_path}")
        return task

    @staticmethod
    def write_url_to_xmp_metadata(image_path, file_url, page_url=None):
        if pyexiv2 is None:
            raise ModuleNotFoundError(
                "pyexiv2 is required to write XMP metadata. "
                "Install it (and system deps) or use crawl-mode 'urls' (LinkPrinter) which doesn't write XMP."
            )
        with pyexiv2.Image(image_path) as img:
            url_data = {'file_url': file_url, "page_url": page_url}
            img.modify_xmp({'Xmp.dc.source': json.dumps(url_data)})

class SourceBingParser(BingParser):
    """ICrawler parser component for Bing that saves image source webpage URLs (URL of containing site, not only direct image URL)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_page_url(self, soup, href_str):
        page_url = None
        lnkw = soup.find("div", class_="lnkw")
        if lnkw and lnkw.a:
            page_url = lnkw.a["href"]
        else:
            try:
                info = json.loads(href_str)
                page_url = info.get("purl", None)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return page_url

    def parse(self, response):
        # adapted from icrawler.builtin.BingParser:
        soup = BeautifulSoup(response.content.decode("utf-8", "ignore"), "lxml")
        image_divs = soup.find_all("div", class_="imgpt")
        pattern = re.compile(r"murl\":\"(.*?)\.jpg")
        for div in image_divs:
            try:
                href_str = html.unescape(div.a["m"])
            except KeyError:
                continue
            match = pattern.search(href_str)
            if match:
                name = match.group(1)
                img_url = f"{name}.jpg"
                result_dict = dict(file_url=img_url)
                page_url = self.get_page_url(soup, href_str)
                if page_url:
                    result_dict["page_url"] = page_url
                yield result_dict

class SourcePagePseudoParser(Parser):
    """ICrawler parser component that yields page_url from task as file_url, for use with UrlListCrawler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # copied and adapted from icrawler.builtin.PseudoParser: https://github.com/hellock/icrawler/blob/835863efc0887770547dd8a066b5db8526480daa/icrawler/builtin/urllist.py#L7
    def worker_exec(self, queue_timeout=2, **kwargs):
        while True:
            if self.signal.get("reach_max_num"):
                self.logger.info("downloaded image reached max num, thread %s" " exit", threading.current_thread().name)
                break
            if self.signal.get("exceed_storage_space"):
                self.logger.info(
                    "downloaded image reached max storage space, thread %s" " exit", threading.current_thread().name
                )
                break
            try:
                url_info = json.loads(self.in_queue.get(timeout=queue_timeout))
                file_url = url_info["file_url"]
                page_url = url_info["page_url"]
            except queue.Empty:
                if self.signal.get("feeder_exited"):
                    self.logger.info("no more page urls to parse, thread %s" " exit", threading.current_thread().name)
                    break
                else:
                    self.logger.info("%s is waiting for new page urls", threading.current_thread().name)
                    continue
            except (json.JSONDecodeError, UnicodeDecodeError ) as e:
                self.logger.error("url json malformatted: %s", e)
                continue
            except Exception as e:
                self.logger.error("exception caught in thread %s: %s", threading.current_thread().name, e)
                continue
            else:
                self.logger.debug(f"start downloading page {file_url}")
            self.output({"file_url": file_url, "page_url": page_url})
