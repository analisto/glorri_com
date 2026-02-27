import csv
import json
import time
import warnings
import requests
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

BASE_URL = "https://api.glorri.com/job-service-v2/jobs/public"
LIMIT = 18
OUTPUT_CSV = "data/glorri.csv"
OUTPUT_JSON = "data/glorri.json"

HEADERS = {
    "accept": "*/*",
    "accept-language": "az",
    "content-type": "application/json",
    "origin": "https://jobs.glorri.com",
    "referer": "https://jobs.glorri.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
}


def fetch_page(offset: int) -> dict:
    resp = requests.get(
        BASE_URL,
        params={"offset": offset, "limit": LIMIT},
        headers=HEADERS,
        timeout=30,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()


def flatten_job(job: dict) -> dict:
    """Flatten nested job dict to a single level for CSV."""
    flat = {}
    for key, val in job.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[f"{key}_{sub_key}"] = sub_val
        elif isinstance(val, list):
            flat[key] = json.dumps(val, ensure_ascii=False)
        else:
            flat[key] = val
    return flat


def main():
    print("Fetching first page to detect total count...")
    first = fetch_page(0)

    # Common response shapes: {total, jobs:[...]}, {count, results:[...]}, etc.
    jobs_key = next(
        (k for k in first if isinstance(first[k], list)), None
    )
    if jobs_key is None:
        print("Could not find jobs array in response. Raw keys:", list(first.keys()))
        return

    total = first.get("total") or first.get("count") or first.get("totalCount")
    first_page_jobs = first[jobs_key]

    print(f"Jobs key: '{jobs_key}', Total: {total}, First page count: {len(first_page_jobs)}")

    all_jobs = list(first_page_jobs)

    if total:
        offsets = range(LIMIT, total, LIMIT)
    else:
        # No total — keep fetching until empty page
        offsets = range(LIMIT, 10_000, LIMIT)

    for offset in offsets:
        print(f"  Fetching offset={offset}...")
        data = fetch_page(offset)
        page_jobs = data.get(jobs_key, [])
        if not page_jobs:
            print("  Empty page — done.")
            break
        all_jobs.extend(page_jobs)
        time.sleep(0.3)  # be polite

    print(f"\nTotal jobs collected: {len(all_jobs)}")

    # Save raw JSON
    import os
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_jobs, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON -> {OUTPUT_JSON}")

    # Save CSV
    flat_jobs = [flatten_job(j) for j in all_jobs]
    fieldnames = list({k for job in flat_jobs for k in job})
    fieldnames.sort()

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_jobs)
    print(f"Saved CSV  -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
