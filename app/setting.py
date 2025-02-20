import os

A_VALUE = float(os.getenv("A_VALUE", 1.25))
B_VALUE = float(os.getenv("B_VALUE", 2.5))
REVIEW_WEIGHT = float(os.getenv("REVIEW_WEIGHT", 0.4))
CAUTION_WEIGHT = float(os.getenv("CAUTION_WEIGHT", 0.15))
CONVENIENCE_WEIGHT = float(os.getenv("CONVENIENCE_WEIGHT", 0.15))
