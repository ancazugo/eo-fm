import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project Constants
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
GEE_PROJECT_NAME = os.getenv("GEE_PROJECT_NAME")
PROJECT_CRS = "EPSG:4326"

# LCZ Dictionary
lcz_dict = {1: {"name": "Compact High-Rise", "alt_code": "1", "color": "#8c0000"},
            2: {"name": "Compact Mid-Rise", "alt_code": "2", "color": "#d10000"},
            3: {"name": "Compact Low-Rise", "alt_code": "3", "color": "#ff0000"},
            4: {"name": "Open High-Rise", "alt_code": "4", "color": "#bf4d00"},
            5: {"name": "Open Mid-Rise", "alt_code": "5", "color": "#ff6600"},
            6: {"name": "Open Low-Rise", "alt_code": "6", "color": "#ff9955"},
            7: {"name": "Lightweight Low-Rise", "alt_code": "7", "color": "#faee05"},
            8: {"name": "Large Low-Rise", "alt_code": "8", "color": "#bcbcbc"},
            9: {"name": "Sparsely Built", "alt_code": "9", "color": "#ffccaa"},
            10: {"name": "Heavy Industry", "alt_code": "10", "color": "#555555"},
            11: {"name": "Dense Trees", "alt_code": "A", "color": "#006a00"},
            12: {"name": "Scattered Trees", "alt_code": "B", "color": "#00aa00"},
            13: {"name": "Bush, Scrub", "alt_code": "C", "color": "#648525"},
            14: {"name": "Low Plants", "alt_code": "D", "color": "#b9db79"},
            15: {"name": "Bare Rock or Paved", "alt_code": "E", "color": "#000000"},
            16: {"name": "Bare Soil or Sand", "alt_code": "F", "color": "#fbf7ae"},
            17: {"name": "Water", "alt_code": "G", "color": "#6a6aff"}}

# Koppen-Geiger Dictionary
koppen_dict = {1: {"name": "Af", "description": "Tropical, rainforest", "color": [0, 0, 255]},
               2: {"name": "Am", "description": "Tropical, monsoon", "color": [0, 120, 255]},
               3: {"name": "Aw", "description": "Tropical, savannah", "color": [70, 170, 250]},
               4: {"name": "BWh", "description": "Arid, desert, hot", "color": [255, 0, 0]},
               5: {"name": "BWk", "description": "Arid, desert, cold", "color": [255, 150, 150]},
               6: {"name": "BSh", "description": "Arid, steppe, hot", "color": [245, 165, 0]},
               7: {"name": "BSk", "description": "Arid, steppe, cold", "color": [255, 220, 100]},
               8: {"name": "Csa", "description": "Temperate, dry summer, hot summer", "color": [255, 255, 0]},
               9: {"name": "Csb", "description": "Temperate, dry summer, warm summer", "color": [200, 200, 0]},
               10: {"name": "Csc", "description": "Temperate, dry summer, cold summer", "color": [150, 150, 0]},
               11: {"name": "Cwa", "description": "Temperate, dry winter, hot summer", "color": [150, 255, 150]},
               12: {"name": "Cwb", "description": "Temperate, dry winter, warm summer", "color": [100, 200, 100]},
               13: {"name": "Cwc", "description": "Temperate, dry winter, cold summer", "color": [50, 150, 50]},
               14: {"name": "Cfa", "description": "Temperate, no dry season, hot summer", "color": [200, 255, 80]},
               15: {"name": "Cfb", "description": "Temperate, no dry season, warm summer", "color": [100, 255, 80]},
               16: {"name": "Cfc", "description": "Temperate, no dry season, cold summer", "color": [50, 200, 0]},
               17: {"name": "Dsa", "description": "Cold, dry summer, hot summer", "color": [255, 0, 255]},
               18: {"name": "Dsb", "description": "Cold, dry summer, warm summer", "color": [200, 0, 200]},
               19: {"name": "Dsc", "description": "Cold, dry summer, cold summer", "color": [150, 50, 150]},
               20: {"name": "Dsd", "description": "Cold, dry summer, very cold winter", "color": [150, 100, 150]},
               21: {"name": "Dwa", "description": "Cold, dry winter, hot summer", "color": [170, 175, 255]},
               22: {"name": "Dwb", "description": "Cold, dry winter, warm summer", "color": [90, 120, 220]},
               23: {"name": "Dwc", "description": "Cold, dry winter, cold summer", "color": [75, 80, 180]},
               24: {"name": "Dwd", "description": "Cold, dry winter, very cold winter", "color": [50, 0, 135]},
               25: {"name": "Dfa", "description": "Cold, no dry season, hot summer", "color": [0, 255, 255]},
               26: {"name": "Dfb", "description": "Cold, no dry season, warm summer", "color": [55, 200, 255]},
               27: {"name": "Dfc", "description": "Cold, no dry season, cold summer", "color": [0, 125, 125]},
               28: {"name": "Dfd", "description": "Cold, no dry season, very cold winter", "color": [0, 70, 95]},
               29: {"name": "ET", "description": "Polar, tundra", "color": [178, 178, 178]},
               30: {"name": "EF", "description": "Polar, frost", "color": [102, 102, 102]}}
