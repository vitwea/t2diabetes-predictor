import os
import sys
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger("download_cycle")

BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"
CYCLES = [
    "2011-2012", 
    "2013-2014", 
    "2015-2016", 
    "2017-2020", 
    "2021-2023"
]
DOWNLOAD_DIR = "data/nhanes_data/raw"

# ========================================
# SPECIFIC VARIABLES TO DOWNLOAD
# ========================================
TARGET_VARIABLES = {
    # DEMO - Demographics (RIDAGEYR, RIAGENDR, RIDRETH1, INDFMPIR)
    "DEMO": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR"],

    # BMX - Body Measures (BMXBMI, BMXWAIST, BMXWT, BMXHT)
    "BMX": ["BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"],

    # BPX - Blood Pressure (BPXSY1, BPXDI1)
    "BPX": ["BPXSY1", "BPXDI1"],

    # BPQ - Blood Pressure Questionnaire - Hipertension (BPQ020)
    "BPQ": ["BPQ020"],
    
    # TRIGLY - Triglycerides (LBXTR)
    "TRIGLY": ["LBXTR","LBXTLG"],
    
    # HDL - HDL Cholesterol (LBDHDD)
    "HDL": ["LBDHDD"],
    
    # TCHOL - Total Cholesterol (LBXTC)
    "TCHOL": ["LBXTC"],
    
    # BIOPRO - Biochemistry Profile (LBXALT, LBXSCR)
    "BIOPRO": ["LBXALT", "LBXSCR"],
    
    # MCQ - Medical Conditions
    "MCQ": ["MCQ160L", "MCQ160E"],
    
    # SMQ - Smoking (SMQ020)
    "SMQ": ["SMQ040"],
    
    # SLD - Sleep hours (SLD010H)
    "SLQ": ["SLD10H","SLD012"],
    
    # GHB - Glucose (LBXGH - TARGET VARIABLE)
    "GHB": ["LBXGH"]   
}

def sanitize_filename(name):
    """Remove invalid characters from filenames"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def get_xpt_links(filecode, cycle):
    """Fetch XPT download links for a specific file code and cycle"""
    url = f"{BASE_URL}?Cycle={cycle}"
    try:
        logger.debug(f"Fetching links for {filecode} {cycle}")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            if href.endswith('.xpt') and filecode.lower() in href:
                full_link = f"https://wwwn.cdc.gov{a['href']}"
                links.append(full_link)
                logger.debug(f"Found: {full_link}")
        
        if links:
            logger.info(f"Found {len(links)} files for {filecode}")
        else:
            logger.warning(f"No files found for {filecode} in {cycle}")
        return links
    except Exception as e:
        logger.error(f"Error fetching links for {filecode} {cycle}: {e}")
        return []

def download_file(url, folder):
    """Download XPT file from CDC"""
    filename = sanitize_filename(os.path.basename(url))
    filepath = os.path.join(folder, filename)
    
    # Skip if already exists
    if os.path.exists(filepath):
        logger.info(f"Already exists: {filename}")
        return filepath
    
    try:
        logger.info(f"DOWNLOADING {filename}...")
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        size_mb = total_size / 1024 / 1024
        logger.info(f"Downloaded {filename}: {size_mb:.2f} MB")
        return filepath
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return None

def main():
    logger.info("="*70)
    logger.info("DOWNLOADING")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
    logger.info(f"Total cycles: {len(CYCLES)}")
    logger.info(f"Total file types: {len(TARGET_VARIABLES)}")
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    total_files_downloaded = 0
    total_files_skipped = 0
    
    for cycle_idx, cycle in enumerate(CYCLES, 1):
        logger.info(f"\nCycle {cycle_idx}/{len(CYCLES)}: Processing {cycle}")
        cycle_folder = os.path.join(DOWNLOAD_DIR, cycle)
        os.makedirs(cycle_folder, exist_ok=True)
        
        files_in_cycle = 0
        skipped_in_cycle = 0
        
        for filecode, variables in TARGET_VARIABLES.items():
            logger.info(f"  -> {filecode} ({', '.join(variables[:3])}{'...' if len(variables) > 3 else ''})")
            links = get_xpt_links(filecode, cycle)
            
            if not links:
                logger.warning(f"    No links found for {filecode}")
                continue
                
            for link in links:
                result = download_file(link, cycle_folder)
                if result:
                    # Check if file was newly downloaded (not just skipped)
                    if os.path.getmtime(result) > (datetime.now().timestamp() - 10):
                        files_in_cycle += 1
                        total_files_downloaded += 1
                    else:
                        skipped_in_cycle += 1
                        total_files_skipped += 1
        
        logger.info(f"Cycle summary: {files_in_cycle} downloaded, {skipped_in_cycle} skipped")
    
    logger.info("\n" + "="*70)
    logger.info("Download completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total files downloaded: {total_files_downloaded}")
    logger.info(f"Total files skipped (already existed): {total_files_skipped}")
    logger.info(f"Output directory: {os.path.abspath(DOWNLOAD_DIR)}")
    logger.info("="*70)
    logger.info("Log saved to nhanes_download.log")

if __name__ == "__main__":
    main()
