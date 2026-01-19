import os
import requests
from bs4 import BeautifulSoup
import re

BASE_URL = "https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx"

CYCLES = [
    "2005-2006","2007-2008","2009-2010","2011-2012",
    "2013-2014","2015-2016","2017-2018","2019-2020",
    "2021-2022"
]

DOWNLOAD_DIR = "./data/nhanes_data"

# ---------------------------
# Sanitizar nombres
# ---------------------------
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# ---------------------------
# Obtener enlaces XPT por ciclo
# ---------------------------
def get_xpt_links_by_filename(file_code, cycle):
    url = f"{BASE_URL}?Cycle={cycle}"
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith(".xpt") and file_code.lower() in href:
            links.append("https://wwwn.cdc.gov" + a["href"])
    return links

# ---------------------------
# Descargar fichero
# ---------------------------
def download_file(url, folder):
    filename = sanitize_filename(os.path.basename(url))
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        print(f"Ya existe: {filepath}")
        return

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Descargado: {filepath}")
    except Exception as e:
        print(f"Error descargando {url}: {e}")

# ---------------------------
# Lógica ciclo-dependiente
# ---------------------------
CYCLES_DG = ["2003-2004","2005-2006","2007-2008","2009-2010","2011-2012"]
CYCLES_HJ = ["2013-2014","2015-2016","2017-2018"]

FILE_MAP = {
    "SEQN": ["DEMO"],
    "RIDAGEYR": ["DEMO"],
    "RIAGENDR": ["DEMO"],
    "RIDRETH1": ["DEMO"],
    "DIQ010": ["DIQ"],
    "DIQ160": ["DIQ"],
    "LBXGLU": ["GLU"],
    "LBXIN": ["GLU", "INS"],
    "LBXGH": ["GLU", "GHB"],
    "BMXHT": ["BMX"],
    "BMXBMI": ["BMX"],
    "BMXWAIST": ["BMX"],
    "BPXSY1": ["BPX"],
    "BPXSY2": ["BPX"],
    "BPXDI1": ["BPX"],
    "BPXDI2": ["BPX"],
    "DR1TPROT": ["DR1TOT"],
    "DR1TCARB": ["DR1TOT"],
    "DR1TTFAT": ["DR1TOT"],
    "SMQ020": ["SMQ"],
    "LBXTC": ["TCHOL"],
    "LBDHDL": ["HDL"],
    "LBXTR": ["TRIGLY"],
    "LBDLDL": ["LDL"],
    "LBXGGT": ["BIOPRO"],
    "LBXALT": ["BIOPRO"],
    "LBXSCR": ["BIOPRO"],
    "LBXUAPB": ["ALB_CR"],
    "LBXCRP": ["CRP"],
}

def files_for_cycle(variable, cycle):
    if variable in ["LBXGLU", "LBXIN", "LBXGH"]:
        if cycle in CYCLES_DG:
            return ["GLU"]
        elif cycle in CYCLES_HJ:
            if variable == "LBXGLU":
                return ["GLU"]
            if variable == "LBXIN":
                return ["INS"]
            if variable == "LBXGH":
                return ["GHB"]
        else:
            if variable == "LBXGLU":
                return ["GLU"]
            if variable == "LBXIN":
                return ["INS"]
            if variable == "LBXGH":
                return ["GHB"]
    return FILE_MAP[variable]

# ---------------------------
# Variables objetivo
# ---------------------------
VARIABLES = list(FILE_MAP.keys())

# ---------------------------
# Main
# ---------------------------
def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for cycle in CYCLES:
        print(f"\n=== Ciclo {cycle} ===")

        cycle_folder = os.path.join(DOWNLOAD_DIR, cycle)
        os.makedirs(cycle_folder, exist_ok=True)

        needed_files = set()

        for var in VARIABLES:
            needed_files.update(files_for_cycle(var, cycle))

        print(f"Ficheros necesarios: {sorted(needed_files)}")

        for file_code in needed_files:
            links = get_xpt_links_by_filename(file_code, cycle)
            for link in links:
                download_file(link, cycle_folder)

if __name__ == "__main__":
    main()