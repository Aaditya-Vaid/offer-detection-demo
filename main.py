from __future__ import annotations
import csv
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
import os
from urllib.parse import urljoin
import time
import base64
from dotenv import load_dotenv
import easyocr
import requests
import uuid
from typing import List
from pydantic import BaseModel, Field, ConfigDict
import logging
from openai import AzureOpenAI
from datetime import datetime
import re


CSV_FILE_PATH = Path(
    r"paste\your\path\here\your_csv_file.csv"
)
JSON_FILE_PATH = CSV_FILE_PATH.with_suffix(".json")

# retry configuration
MAX_ATTEMPTS = 5
MAX_DELAY_SECONDS = 60
INITIAL_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60

# load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    filename="brands.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)


# OCR class
class CustomOCR:
    def __init__(self, images: List[dict], retries=3, timeout=10):
        self.images = images
        self.retries = retries
        self.timeout = timeout
        self.reader = easyocr.Reader(["en"])
        self.result = []

    # ---------- normalize URL ----------
    def normalize_url(self, base_url, src):

        if not src:
            return ""

        src = src.strip()

        if src.startswith("//"):
            src = "https:" + src

        src = urljoin(base_url, src)

        return src

    # ---------- download + OCR ----------
    def process_image(self, url, base_url):

        # standardize URL
        url = self.normalize_url(base_url, url)

        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"

        try:
            for attempt in range(self.retries):
                try:
                    if url.startswith("data:image"):
                        header, encoded = url.split(",", 1)
                        image_data = base64.b64decode(encoded)

                        with open(temp_filename, "wb") as f:
                            f.write(image_data)

                    else:
                        headers = {"User-Agent": "Mozilla/5.0", "Referer": base_url}

                        response = requests.get(
                            url, headers=headers, timeout=self.timeout
                        )

                        response.raise_for_status()

                        with open(temp_filename, "wb") as f:
                            f.write(response.content)

                    # ---------- OCR ----------
                    result = self.reader.readtext(temp_filename)

                    text = " ".join([word[1] for word in result])

                    return text

                except Exception as e:
                    # print(f"Retry {attempt + 1}")
                    # logging.warning(f"Retry {attempt + 1} failed for {url}")

                    if attempt < self.retries - 1:
                        time.sleep(1)
                    else:
                        # print(f"Failed after {self.retries} attempts: {url}")
                        return ""

        finally:
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except PermissionError:
                # logging.warning(f"Could not delete temp file: {temp_filename}")
                pass

    # ---------- run OCR ----------
    def run_ocr(self):

        for count, image in enumerate(self.images, start=1):
            text = self.process_image(image["src"], image["brand_url"])

            self.result.append(
                {
                    "brand_name": image["brand_name"],
                    "brand_url": image["brand_url"],
                    "Chain Id": image["Chain Id"],
                    "src": self.normalize_url(image["brand_url"], image["src"]),
                    "alt": image["alt"],
                    "text": text,
                }
            )

            if count % 5 == 0:
                print(count)

        return self.result


class OffersClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_offer: bool = Field(
        description="True if clear promotional text (discount, sale, offer) is present. Otherwise False."
    )


# Pydantic data validation class
class OffersAvailable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_offer: bool = Field(
        description="True if clear promotional text (discount, sale, offer) is present. Otherwise False."
    )

    title: str = Field(
        description=(
            "Short readable title ONLY if is_offer is true. "
            "Include key info like discount or product. "
            "Return empty string if is_offer is false."
        ),
        max_length=100,
    )
    validity_date: str = Field(
        description=(
            "Offer expiry date in DD/MM/YYYY format ONLY if is_offer is true. "
            "Return empty string if is_offer is false or expiry date not clear."
        )
    )


# Parent class for OllamaAgent and OpenAIAgent
class ParentAgent:
    def __init__(self, model, ocrTexts, time_delay):
        self.inputs = ocrTexts
        self.model = model
        self.time_delay = time_delay  # retry after this time.
        self.total_tokens = 0

    def call_llm(self, prompt):
        """Using polymorphism for two different usage of this method."""
        return {}

    def call_classification_model(self, prompt):
        return {}

    def run_classification_agent(self):
        prompt = """
        Your task is to decide whether the given text extracted from an image via ocr contains a offer or not. If it contains an offer then return is_offer as true otherwise false.
        Remember that any text that contains a numerical value is not an offer. Offers usually contain words like Upto, UP TO,  discount, sale, offer, save x%, get upto x% off, flat x discount, UP TO x% discount, buy one get one free, free delivery on purchase of x, starting at x etc. If the text contains any such promotional language then it is an offer. If the text is vague and does not clearly indicate a offer, then return false. Always be strict in your classification and only classify as offer when you are sure that the text contains a offer.
        Here is the raw text given
        raw text - "{raw_text}"
        """

        result = []
        for count, row in enumerate(self.inputs, start=1):
            output, response = self.call_classification_model(
                prompt=prompt.format(raw_text=row["text"])
            )
            try:
                usage = response.usage.total_tokens
                self.total_tokens += usage
            except Exception as e:
                usage = ""

            res = {
                "brand_name": row["brand_name"],
                "brand_url": row["brand_url"],
                "chain_id": row["Chain Id"],
                "image_url": row["src"],
                "is_offer": output.is_offer,
                "alt_text": row["alt"],
                "ocr_text": row["text"],
                "usage": usage,
            }
            result.append(res)
            if count % 5 == 0:
                print(count)
        return result, self.total_tokens

    def run_agent(self, inputs):
        prompt = """Given is the raw ocr text extracted from an image and it is confirmed to have an offer. Generate one sentence Title that attracts audience. I am providing you with raw data extracted from OCR from an image extract. If the company name is unknown, then only show the offer.
        Here is the raw text given raw text - "{raw_text}" Do not use the brand name in the output - "{brand}". If the offer details are not clear in the raw text, then return empty string as title. Always return a title that is less than 100 characters.
        
        """
        result = []
        for count, row in enumerate(inputs, start=1):
            if not row["is_offer"]:
                res = {
                    "brand_name": row["brand_name"],
                    "brand_url": row["brand_url"],
                    "chain_id": row["chain_id"],
                    "image_url": row["image_url"],
                    "is_offer": row["is_offer"],
                    "alt_text": row["alt_text"],
                    "ocr_text": row["ocr_text"],
                    "title": "",
                    "date": "",
                    "usage": "",
                }
                result.append(res)
            else:
                output, response = self.call_llm(
                    prompt=prompt.format(
                        raw_text=row["ocr_text"], brand=row["brand_name"]
                    )
                )
                try:
                    usage = response.usage.total_tokens
                    self.total_tokens += usage
                except Exception as e:
                    usage = ""

                res = {
                    "brand_name": row["brand_name"],
                    "brand_url": row["brand_url"],
                    "chain_id": row["chain_id"],
                    "image_url": row["image_url"],
                    "is_offer": output.is_offer,
                    "alt_text": row["alt_text"],
                    "ocr_text": row["ocr_text"],
                    "title": output.title,
                    "date": output.validity_date,
                    "usage": usage,
                }
                result.append(res)
                if count % 5 == 0:
                    print(count)
        return result, self.total_tokens


# OpenAIAgent class: child class of ParentAgent
class OpenAIAgent(ParentAgent):
    def __init__(self, model, ocrTexts, time_delay=65):
        super().__init__(model, ocrTexts, time_delay=65)
        self.client = AzureOpenAI(
            api_key=os.getenv("azure_openai_api_key"),
            azure_endpoint=os.getenv("azure_endpoint"),
            api_version=os.getenv("azure_openai_api_version"),
        )

    def call_classification_model(self, prompt):
        conversations = [{"role": "system", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversations,
                temperature=0.5,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "OffersClassification",
                        "schema": OffersClassification.model_json_schema(),
                    },
                },
            )
            gpt_response = response.choices[0].message.content
            validated_response = OffersClassification.model_validate(
                json.loads(gpt_response)
            )
            return validated_response, response
        except Exception as err:
            # logging.exception(
            #     f"Error calling GPT LLM client for classification: {err}. Retrying in {self.time_delay} seconds..."
            # )
            time.sleep(self.time_delay)
            # Retry once
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=conversations,
                    temperature=0.5,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "OffersClassification",
                            "schema": OffersClassification.model_json_schema(),
                        },
                    },
                )

                gpt_response = response.choices[0].message.content
                validated_response = OffersClassification.model_validate(
                    json.loads(gpt_response)
                )
                return validated_response, response

            except Exception as err2:
                # logging.exception(f"Retry also failed for classification: {err2}")
                fallback = OffersClassification(is_offer=False)
                return fallback, ""

    def call_llm(self, prompt):
        """
        Calls Azure GPT LLM client and handles retries.
        """

        conversations = [{"role": "system", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversations,
                temperature=0.5,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "OffersAvailable",
                        "schema": OffersAvailable.model_json_schema(),
                    },
                },
            )
            gpt_response = response.choices[0].message.content
            validated_response = OffersAvailable.model_validate(
                json.loads(gpt_response)
            )
            return validated_response, response
        except Exception as err:
            # logging.exception(
            #     f"Error calling GPT LLM client: {err}. Retrying in {self.time_delay} seconds..."
            # )
            time.sleep(self.time_delay)
            # Retry once
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=conversations,
                    temperature=0.5,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "OffersAvailable",
                            "schema": OffersAvailable.model_json_schema(),
                        },
                    },
                )

                gpt_response = response.choices[0].message.content
                validated_response = OffersAvailable.model_validate(
                    json.loads(gpt_response)
                )
                return validated_response, response

            except Exception as err2:
                # logging.exception(f"Retry also failed: {err2}")
                fallback = OffersAvailable(
                    is_offer=False,
                    title="",
                    validity_date="",
                )
                return fallback, ""


def csv_to_json(csv_path: Path, json_path: Path) -> None:
    """Convert a CSV file into JSON using the first row as column names."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        reader.fieldnames = [fieldname.strip() for fieldname in reader.fieldnames]

        rows = list(reader)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(rows, json_file, indent=4, ensure_ascii=False)


def get_image_url(item):
    """
    Return best usable image URL from src or srcset
    """
    src = item.get("src")

    if src:
        return src

    srcset = item.get("srcset")
    if srcset:
        # take first URL from srcset
        return srcset.split(",")[0].split(" ")[0]

    return None


def valid_image(url):
    if not url:
        return False

    banned = [".svg", ".webp", ".gif"]

    url = url.lower()

    for b in banned:
        if b in url:
            return False

    return True


def normalize_url(base_url, src):
    if not src:
        return ""

    src = src.strip()

    # protocol relative URLs
    if src.startswith("//"):
        src = "https:" + src

    # convert relative → absolute
    src = urljoin(base_url, src)

    return src


def extract_srcset(srcset, base_url):

    urls = []

    if srcset:
        parts = srcset.split(",")

        for p in parts:
            url = p.strip().split(" ")[0]
            url = normalize_url(base_url, url)
            urls.append(url)

    return urls


def safe_get(driver, url, retries=3):

    for attempt in range(retries):
        try:
            driver.get(url)
            return True
        except WebDriverException:
            print(f"Retry {attempt + 1} loading {url}")
            time.sleep(3)

    return False


def extract_images(brand_name, brand_url, chain_id):

    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)

    images = []

    try:
        if not safe_get(driver, brand_url):
            print(f"❌ Failed to load {brand_url}")
            return

        try:
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
        except TimeoutException:
            print("⚠️ Page load timeout — capturing partial content")

        time.sleep(10)  # allow some extra time for images to load
        img_elements = driver.find_elements(By.TAG_NAME, "img")

        for img in img_elements:
            try:
                alt = img.get_attribute("alt") or ""

                # ---------- IMG src ----------
                src = img.get_attribute("src")

                src = normalize_url(brand_url, src)

                if valid_image(src):
                    images.append(
                        {
                            "brand_name": brand_name,
                            "brand_url": brand_url,
                            "Chain Id": chain_id,
                            "tag": "img",
                            "src": src,
                            "alt": alt,
                        }
                    )

                # ---------- IMG srcset ----------
                srcset = img.get_attribute("srcset")

                urls = extract_srcset(srcset, brand_url)

                for u in urls:
                    if valid_image(u):
                        images.append(
                            {
                                "brand_name": brand_name,
                                "brand_url": brand_url,
                                "Chain Id": chain_id,
                                "tag": "img-srcset",
                                "src": u,
                                "alt": alt,
                            }
                        )

                # ---------- parent picture ----------
                picture = img.find_elements(By.XPATH, "./ancestor::picture")

                if picture:
                    sources = picture[0].find_elements(By.TAG_NAME, "source")

                    for source in sources:
                        srcset = source.get_attribute("srcset")

                        urls = extract_srcset(srcset, brand_url)

                        for u in urls:
                            if valid_image(u):
                                images.append(
                                    {
                                        "brand_name": brand_name,
                                        "brand_url": brand_url,
                                        "Chain Id": chain_id,
                                        "tag": "picture-source",
                                        "src": u,
                                        "alt": alt,
                                    }
                                )

            except Exception:
                continue

    finally:
        driver.quit()

    # ---------- remove duplicates ----------
    seen = set()
    unique_images = []

    for item in images:
        key = item["src"]

        if key not in seen:
            seen.add(key)
            unique_images.append(item)

    # ---------- save JSON ----------
    folder = "Images_new"
    os.makedirs(folder, exist_ok=True)

    file_path = os.path.join(folder, f"{brand_name}_images.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(unique_images, f, ensure_ascii=False, indent=4)

    print(f"JSON saved as {brand_name}_images.json")
    print(f"Total valid images: {len(unique_images)}")


def main() -> None:
    csv_to_json(CSV_FILE_PATH, JSON_FILE_PATH)
    print(f"JSON saved to {JSON_FILE_PATH}")

    # load the JSON and iterate the list under the top-level 'brands' key
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(data)

    for brand in data[:]:
        name = brand.get("Brand Name", "").strip()
        url = brand.get("Website", "").strip()
        chain_id = brand.get("Chain Id", "")

        if not url:
            print(f"Skipping {name!r}: missing Website URL")
            continue

        print(f"Extracting images for {name} from {url}...")
        extract_images(name, url, chain_id)

    folder_path = r"Images_new"

    # Only files directly inside the folder (not subfolders)
    files = [str(p) for p in Path(folder_path).iterdir() if p.is_file()]

    print(f"Total files: {len(files)}")
    for file in files:
        print(f"Processing file: {file}")
        start_time = time.time()
        with open(file, "r", encoding="utf-8") as f:
            images = json.load(f)
        f_path = Path(file)
        brand_name = re.sub(r"_images$", "", f_path.stem, flags=re.IGNORECASE)

        print(f"Total images extracted: {len(images)}")

        # removing icon images and web images: filter: .svg and .webp
        final_images = []

        for i in images:
            url = get_image_url(i)

            if not url:
                continue

            if any(ext in url for ext in [".svg", ".webp", ".gif"]):
                continue

            i["src"] = url
            final_images.append(i)
        print(f"Without icon images: {len(final_images)}")

        # Removing dupticates data points
        seen = set()
        unique_items = []
        for item in final_images:
            key = (item["alt"], item["src"])  # unique identity

            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        print(f"Unique images: {len(unique_items)}")

        # Initialize OCR object
        ocr = CustomOCR(
            images=unique_items,
        )
        ocr_images = (
            ocr.run_ocr()
        )  # running ocr pipeline: outputs images with raw ocr text
        # Removing images with characters less than 5 characters in the raw ocr text
        keep_text_only = []
        for text in ocr_images:
            if len(text["text"]) > 5:
                keep_text_only.append(text)
            else:
                pass
        print(
            f"Images after removing text less than 5 characters: {len(keep_text_only)}"
        )
        agent = OpenAIAgent(model="gpt-4o-mini", ocrTexts=keep_text_only)
        offers, tokens_used = agent.run_classification_agent()
        offers, tokens_used = agent.run_agent(
            inputs=offers
        )  # running classification pipeline: outputs structured dictionary
        end_time = time.time()
        time_taken = end_time - start_time
        # Save as csv
        rows = offers
        try:
            fieldnames = [key for key in rows[0].keys()]
            folder = "Offers_csv_new"
            csv_filename = f"{brand_name}_offers_gpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            file_path = os.path.join(folder, csv_filename)
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"CSV saved as {csv_filename}")
            print(f"completed in {time_taken} seconds")
            folder = "Offers_json_new"

            json_filename = f"{brand_name}_offers_gpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = os.path.join(folder, json_filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=4)
            print(f"JSON saved as {json_filename}")

            folder = "gpt_summary"
            filename = (
                f"{brand_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            result = {
                "json_file_name": json_filename,
                "time_taken": time_taken,
                "total_tokens": tokens_used,
            }
            file_path = os.path.join(folder, filename)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(f"No offers extracted from {brand_name}, JSON not saved.")
            logging.exception(f"Error saving JSON for {brand_name}: {e}")


if __name__ == "__main__":
    main()
