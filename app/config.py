# app/config.py
from decouple import config

A_VALUE = float(config('A_VALUE', default=1.25))
B_VALUE = float(config('B_VALUE', default=2.5))
REVIEW_WEIGHT = float(config('REVIEW_WEIGHT', default=0.4))
CAUTION_WEIGHT = float(config('CAUTION_WEIGHT', default=0.15))
CONVENIENCE_WEIGHT = float(config('CONVENIENCE_WEIGHT', default=0.15))
