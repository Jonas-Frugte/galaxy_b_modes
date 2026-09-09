import sys

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.data_download_soap import download_soap

# python3 data_download_SOAP.py [box_name] [cat_name]
BOX_NAME = sys.argv[1] if len(sys.argv) > 1 else "L2p8_m9"
CAT_NAME = sys.argv[2] if len(sys.argv) > 2 else "real_cat_1"

download_soap(FilePaths(BOX_NAME=BOX_NAME, CAT_NAME=CAT_NAME))
