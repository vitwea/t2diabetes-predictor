import os
import sys
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

from src.utils.logger import get_logger 

# ===========================
# Configure Logging
# ===========================
logger = get_logger("nhanes_download")


# ===========================
# Configuration
# ===========================
BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"

# NHANES cycles with standardized methods (2011+)
CYCLES = ["2011-2012", "2013-2014", "2015-2016", "2017-2020", "2021-2022"]

# Base files for diabetes detection - all compatible across 2011+ cycles
# LBXGLU, LBXGH, LBXSCR are fully standardized (no calibration required)
BASE_FILES = ["DEMO", "DIQ", "GLU", "INS", "GHB", "BMX", "BPX", "TCHOL", "HDL", "TRIGLY", "BIOPRO", "ALB_CR", "SMQ", "DR1TOT"]

DOWNLOAD_DIR = "./data/nhanes_data"

# ===========================
# Sanitize filenames
# ===========================
def sanitize_filename(name):
    """Remove invalid characters from filenames"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# ===========================
# Fetch XPT file links from CDC
# ===========================
def get_xpt_links(file_code, cycle):
    """
    Fetch XPT download links for a specific file code and cycle
    
    Args:
        file_code: NHANES file code (e.g., 'DEMO', 'GLU', 'DIQ')
        cycle: NHANES cycle (e.g., '2011-2012')
    
    Returns:
        List of download URLs for XPT files
    """
    url = f"{BASE_URL}?Cycle={cycle}"
    try:
        logger.debug(f"Fetching links for {file_code} ({cycle})")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if href.endswith(".xpt") and file_code.lower() in href:
                full_link = "https://wwwn.cdc.gov" + a["href"]
                links.append(full_link)
                logger.debug(f"    Found: {full_link}")
        
        if links:
            logger.info(f"      Found {len(links)} file(s) for {file_code}")
        else:
            logger.warning(f"       No files found for {file_code} in {cycle}")
        
        return links
        
    except Exception as e:
        logger.error(f"Error fetching links for {file_code} ({cycle}): {e}")
        return []

# ===========================
# Download file from URL
# ===========================
def download_file(url, folder):
    """
    Download XPT file from CDC
    
    Args:
        url: Full download URL
        folder: Destination folder path
    
    Returns:
        File path if successful, None otherwise
    """
    filename = sanitize_filename(os.path.basename(url))
    filepath = os.path.join(folder, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        logger.info(f"      Already exists: {filename}")
        return filepath
    
    try:
        logger.info(f"      [DOWNLOAD] {filename}...")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        
        total_size = 0
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        size_mb = total_size / (1024 * 1024)
        logger.info(f"       Downloaded {filename} ({size_mb:.2f} MB)")
        return filepath
        
    except Exception as e:
        logger.error(f"      Failed to download {filename}: {e}")
        return None

# ===========================
# Main execution
# ===========================
def main():
    logger.info("=" * 70)
    logger.info("NHANES DATA DOWNLOAD - Cycles 2011-2022")
    logger.info("Standardized Methods (No Calibration Required)")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
    logger.info(f"Total cycles: {len(CYCLES)}")
    logger.info(f"Total file types: {len(BASE_FILES)}")
    logger.info("")
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    total_files_downloaded = 0
    total_files_skipped = 0
    
    for cycle_idx, cycle in enumerate(CYCLES, 1):
        logger.info(f"[Cycle {cycle_idx}/{len(CYCLES)}] Processing: {cycle}")
        
        cycle_folder = os.path.join(DOWNLOAD_DIR, cycle)
        os.makedirs(cycle_folder, exist_ok=True)
        
        files_in_cycle = 0
        skipped_in_cycle = 0
        
        for file_code in BASE_FILES:
            logger.info(f"  [{file_code}]")
            
            links = get_xpt_links(file_code, cycle)
            
            if not links:
                continue
            
            for link in links:
                result = download_file(link, cycle_folder)
                if result:
                    # Check if file was skipped or newly downloaded
                    if os.path.getmtime(result) > (datetime.now().timestamp() - 10):
                        files_in_cycle += 1
                        total_files_downloaded += 1
                    else:
                        skipped_in_cycle += 1
                        total_files_skipped += 1
        
        logger.info(f"  Cycle summary: {files_in_cycle} downloaded, {skipped_in_cycle} skipped\n")
    
    logger.info("=" * 70)
    logger.info("Download completed")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total files downloaded: {total_files_downloaded}")
    logger.info(f"Total files skipped (already existed): {total_files_skipped}")
    logger.info(f"Output directory: {os.path.abspath(DOWNLOAD_DIR)}")
    logger.info("=" * 70)
    logger.info("Log saved to: nhanes_download.log\n")

if __name__ == "__main__":
    main()