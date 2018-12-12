import sys
import os
import time
import datetime
from os import path
import subprocess
import time
from collections import deque
import numpy as np
import random
import tensorflow as tf
import pandas

import psycopg2


# Query
q_path = "/home/xuanhe/join0-order-benchmark/"
q_name = "19c.sql"


# Incremental: 10 -> 100
inc = 5

# table size
tbl_size = {"aka_name": 901343, "char_name" : 3140339, "cast_info" : 36244344, "company_name" : 234997, "info_type" : 113,
            "movie_companies" : 2609129, "movie_info" : 14835720, "name" : 4167491, "role_type" : 12, "title" : 2528312}

conn = psycopg2.connect("dbname=imdbload user=postgres password=postgres")

percent = 10
while percent <= 100:
    for tbl in tbl_size:
        size = tbl_size[tbl] * percent  # where id between 1 and percent




    percent = percent + inc

