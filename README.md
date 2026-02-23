# Epstinefiles

Download DOJ Epstein disclosure PDFs (data sets 1–5) locally and prepare page images for Vision Transformer (ViT) pipelines.

## Setup

```bash
pip install -r requirements.txt
```

## Downloading PDFs

Run the downloader to fetch all PDFs from the [DOJ Epstein Library](https://www.justice.gov/epstein/doj-disclosures/data-set-1-files) (data sets 1–5) into folders `dataset_1/` … `dataset_5/` with original filenames (e.g. `EFTA00000002.pdf`):

```bash
python download_doj_pdfs.py
```

- **Output directory:** PDFs are saved under `epstein_pdfs/` by default. Use `-o DIR` to change it:
  ```bash
  python download_doj_pdfs.py -o /path/to/pdfs
  ```
- **Skip existing:** Already-downloaded files are skipped. Use `--no-skip-existing` to re-download.
- **List only:** To only scrape and print the list of PDF URLs (no downloads):
  ```bash
  python download_doj_pdfs.py --list-only
  ```

### Age verification (18+)

The DOJ site requires age verification before serving PDFs. The script tries to pass this automatically via the `/age-verify` flow. If you get HTML instead of PDFs (e.g. "Access Denied" or age-verification pages):

1. In a browser, open [https://www.justice.gov/epstein](https://www.justice.gov/epstein), click **Yes** for the age prompt, then copy your Cookie header (e.g. from DevTools → Network → any request → Request Headers → Cookie).
2. Run the script with that cookie:
   ```bash
   python download_doj_pdfs.py --cookie "name=value; ..."
   ```
   Or set the environment variable:
   ```bash
   export DOJ_COOKIE="name=value; ..."
   python download_doj_pdfs.py
   ```

## PDF → images for Vision Transformer (ViT)

Standard Vision Transformers expect **raster images** (e.g. PNG/JPEG), not raw PDFs. Use the included script to render each PDF page to a PNG:

```bash
python pdf_to_images.py
```

- Reads PDFs from `epstein_pdfs/` (or the base dir you used for the downloader).
- Writes images to `epstein_pdfs/dataset_1_images/` … `epstein_pdfs/dataset_5_images/`.
- Each page is saved as `{EFTA_id}_page0001.png`, `_page0002.png`, etc.
- **Options:**
  - `-d DPI` — resolution (default 150).
  - `-o DIR` — base directory for `dataset_N_images/` (default: same as PDF base dir).

Then point your ViT dataloader at the `dataset_*_images/` folders (one image per page).

## Running the pipeline on dataset 2 (or any dataset)

To index **dataset 2** (or any dataset) with face recognition, captions, and search:

1. **Put PDFs in the right folder**  
   Place dataset 2 PDFs under `epstein_pdfs/dataset_2/` (or set `EPSTEIN_PDFS_DIR` in `.env` to your base dir).

2. **Convert PDFs to images**  
   From the project root:
   ```bash
   python pdf_to_images.py
   ```
   This writes PNGs to `epstein_pdfs/dataset_2_images/` (only dataset dirs that exist are processed).

3. **Run the pipeline**  
   ```bash
   python run_pipeline.py
   ```
   The pipeline discovers all `dataset_*_images/` folders (1–5) and processes every `*_page*.png` it finds. To replace the existing index and re-index everything (including dataset 2), use:
   ```bash
   python run_pipeline.py --overwrite
   ```

Requires `.env` with `TOGETHER_API_KEY` (embeddings + captions). Face recognition uses `my_db/` (see pipeline docs).

## Layout

After downloading and (optionally) rendering to images:

```
epstein_pdfs/
├── dataset_1/          # PDFs (EFTA00000001.pdf, ...)
├── dataset_1_images/   # PNGs per page (for ViT)
├── dataset_2/
├── dataset_2_images/
├── ...
├── dataset_5/
└── dataset_5_images/
```

## Search UI (after running the pipeline)

Once the pipeline has indexed images into LanceDB, run the web search interface:

```bash
streamlit run search_ui.py
```

Then open the URL shown (e.g. http://localhost:8501). You can search by keyword (e.g. "Epstein island", "people at a party"), choose how many results to show, and optionally filter by dataset. Results show the image, caption, source file, page number, and any recognized people.

Requires `TOGETHER_API_KEY` in `.env` (to embed the search query).

### Deploying the Search UI (e.g. Streamlit Cloud)

The LanceDB index is too large for GitHub. To run the app in the cloud:

1. **Build the DB locally** (one time):
   ```bash
   python run_pipeline.py   # creates lancedb/
   ```

2. **Create a zip of the DB** (contents of `lancedb/`, not the folder itself):
   ```bash
   cd lancedb && zip -r ../lancedb.zip . && cd ..
   ```
   The zip must contain `epstein_images.lance/` at the top level.

3. **Upload the zip** to a URL (e.g. Google Drive “share link”, Dropbox, S3 public URL, or a GitHub Release asset). Get a **direct download URL** (for Dropbox/Drive, use “direct link” or “raw” variants so the URL ends in the file and returns the zip).

4. **Set the URL in the host:**
   - **Streamlit Cloud:** App → Settings → Secrets → add:
     ```toml
     LANCEDB_DOWNLOAD_URL = "https://your-direct-url-to/lancedb.zip"
     TOGETHER_API_KEY = "your-key"
     ```
   - **Local / env:** `export LANCEDB_DOWNLOAD_URL="https://..."`

On first run (or after a cold start), the app will download and unzip the DB, then run search. This can take several minutes for a large zip.

## Requirements

- **Downloader:** `requests`, `beautifulsoup4`, `lxml`
- **PDF → images:** `pymupdf` (see `requirements.txt`)
- **Search UI:** `streamlit`
