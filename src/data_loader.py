# src/data_loader.py

import json

def load_business_data(sc, file_path):
    """
    Load business JSON data.
    """
    return sc.textFile(file_path).map(lambda x: json.loads(x))

def load_user_data(sc, file_path):
    """
    Load user JSON data.
    """
    return sc.textFile(file_path).map(lambda x: json.loads(x))

def load_review_data(sc, file_path):
    """
    Load CSV review data, skipping the header.
    """
    rdd = sc.textFile(file_path)
    # Skip header (assumes header is the first line)
    return rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])

def load_validation_data(sc, file_path):
    """
    Load validation CSV data (with header) and return an RDD of lines (skipping header).
    """
    rdd = sc.textFile(file_path)
    return rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
