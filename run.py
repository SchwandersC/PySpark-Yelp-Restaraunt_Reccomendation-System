# run.py

import sys
import time
import numpy as np
from pyspark import SparkContext

from src.data_loader import load_business_data, load_user_data, load_review_data, load_validation_data
from src.feature_engineering import (
    extract_business_features,
    extract_user_features,
    flatten_features,
    encode_features
)
from src.model import train_model, predict
from src.config import XGB_PARAMS

def main(train_path, val_path, output_path):
    sc = SparkContext('local[*]', 'YelpRatingPrediction')
    start_time = time.time()

    # Load JSON data
    business_rdd = load_business_data(sc, train_path + "business.json")
    user_rdd = load_user_data(sc, train_path + "user.json")

    # Extract features from business and user data
    business_features_rdd = business_rdd.map(extract_business_features)
    user_features_rdd = user_rdd.map(extract_user_features)

    # Create dictionaries keyed by business_id and user_id
    business_features_dict = business_features_rdd.map(lambda x: (x[0], x[1:])).collectAsMap()
    user_features_dict = user_features_rdd.map(lambda x: (x[0], x[1:])).collectAsMap()

    # Broadcast the dictionaries for use in transformations
    broadcast_business = sc.broadcast(business_features_dict)
    broadcast_user = sc.broadcast(user_features_dict)

    # Load and parse review training data
    review_rdd = load_review_data(sc, train_path + "yelp_train.csv")
    parsed_reviews = review_rdd.map(lambda line: line.split(',')) \
                               .map(lambda x: (x[0], x[1], float(x[2])))

    # Combine features for each training record:
    # For each (user_id, business_id, rating), merge user and business features.
    train_rdd = parsed_reviews.map(lambda record: (
        record[0],
        record[1],
        record[2],
        flatten_features(
            broadcast_user.value.get(record[0], []) +
            broadcast_business.value.get(record[1], [])
        )
    ))

    train_data = train_rdd.collect()
    # Separate identifiers, features (X) and target (y)
    user_business_train = np.array([(u, b) for u, b, r, f in train_data])
    X_train = np.array([f for u, b, r, f in train_data])
    y_train = np.array([r for u, b, r, f in train_data])

    # Load and parse validation data (assumed to contain user_id and business_id)
    val_rdd = load_validation_data(sc, val_path)
    parsed_val = val_rdd.map(lambda line: line.split(',')) \
                        .map(lambda x: (x[0], x[1]))
    val_rdd_features = parsed_val.map(lambda record: (
        record[0],
        record[1],
        flatten_features(
            broadcast_user.value.get(record[0], []) +
            broadcast_business.value.get(record[1], [])
        )
    ))

    val_data = val_rdd_features.collect()
    user_business_val = np.array([(u, b) for u, b, f in val_data])
    X_val = np.array([f for u, b, f in val_data])

    # Encode features (impute missing categorical values and apply ordinal encoding)
    X_train_encoded, X_val_encoded = encode_features(X_train, X_val)

    # --------------------------------------------
    # Feature Selection: Select top 15 correlated features
    # --------------------------------------------
    # Compute the correlation between each feature and the target
    correlations = np.array([np.corrcoef(X_train_encoded[:, i], y_train)[0, 1] 
                             for i in range(X_train_encoded.shape[1])])
    
    # Sort feature indices based on the absolute value of correlation (descending order)
    sorted_feature_indices = np.argsort(-np.abs(correlations))
    
    # Select the top 15 features
    top_n = 15
    top_features_indices = sorted_feature_indices[:top_n]
    
    # Print out the top features correlations (for debugging purposes)
    print("Top 15 feature indices and their correlations:")
    for idx in top_features_indices:
        print(f"Feature {idx}: correlation = {correlations[idx]}")
    
    # Reduce training and validation sets to only these top features
    X_train_encoded_top = X_train_encoded[:, top_features_indices]
    X_val_encoded_top = X_val_encoded[:, top_features_indices]
    # --------------------------------------------

    # Train the XGBoost model using only the selected features
    model = train_model(X_train_encoded_top, y_train, XGB_PARAMS)

    # Predict on the validation set
    predictions = predict(model, X_val_encoded_top)

    # Write predictions to output CSV file
    with open(output_path, 'w') as f_out:
        f_out.write("user_id,business_id,prediction\n")
        for (user_id, business_id), pred in zip(user_business_val, predictions):
            f_out.write(f"{user_id},{business_id},{pred}\n")

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} sec")
    sc.stop()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data_path> <val_data_path> <output_file>")
        sys.exit(1)
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    output_path = sys.argv[3]
    main(train_path, val_path, output_path)
