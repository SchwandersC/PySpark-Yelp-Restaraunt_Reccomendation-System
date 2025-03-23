# src/feature_engineering.py

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def extract_business_features(business):
    """
    Extract relevant business features.
    Returns a tuple with business_id and selected attributes.
    """
    if business is None:
        return (None,) * 13  # Adjust the length if needed
    attributes = business.get('attributes', {}) or {}
    return (
        business.get('business_id'),
        business.get('postal_code'),
        business.get('stars'),
        business.get('review_count'),
        business.get('is_open'),
        attributes.get('NoiseLevel'),
        attributes.get('OutdoorSeating'),
        attributes.get('RestaurantsAttire'),
        attributes.get('RestaurantsDelivery'),
        attributes.get('RestaurantsGoodForGroups'),
        attributes.get('RestaurantsPriceRange2'),
        attributes.get('RestaurantsReservations'),
        attributes.get('RestaurantsTakeOut')
    )

def extract_user_features(user):
    """
    Extract relevant user features.
    Returns a tuple with user_id and selected features.
    """
    if user is None:
        return (None, None, None, None)
    return (
        user.get('user_id'),
        user.get('review_count'),
        user.get('yelping_since'),
        user.get('average_stars')
    )

def flatten_features(features):
    """
    Flattens a nested list/tuple of features.
    """
    flat_list = []
    for item in features:
        if isinstance(item, (list, tuple)):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list

def transform_with_default(encoder, data, default=np.nan):
    """
    Transform data using an OrdinalEncoder.
    If an unknown category is found, assigns a default value.
    """
    try:
        return encoder.transform(data)
    except ValueError:
        transformed_data = []
        for row in data:
            transformed_row = []
            for i, item in enumerate(row):
                known_categories = encoder.categories_[i]
                if item in known_categories:
                    transformed_value = np.where(known_categories == item)[0][0]
                else:
                    transformed_value = default
                transformed_row.append(transformed_value)
            transformed_data.append(transformed_row)
        return np.array(transformed_data)

def encode_features(X_train, X_val):
    """
    Impute missing categorical values and apply ordinal encoding.
    Returns encoded training and validation features.
    """
    # Identify indices of categorical features
    categorical_indices = [i for i in range(X_train.shape[1]) if isinstance(X_train[0, i], str)]
    X_train_categorical = X_train[:, categorical_indices]
    X_train_numerical = X_train[:, [i for i in range(X_train.shape[1]) if i not in categorical_indices]].astype(float)

    imputer = SimpleImputer(strategy='constant', fill_value='missing')
    X_train_cat_imp = imputer.fit_transform(X_train_categorical)
    X_train_cat_imp = X_train_cat_imp.astype(str)

    ordinal_encoder = OrdinalEncoder()
    X_train_cat_enc = ordinal_encoder.fit_transform(X_train_cat_imp)

    X_train_encoded = np.hstack([X_train_numerical, X_train_cat_enc])

    # Process validation data using the same imputer and encoder
    X_val_categorical = X_val[:, categorical_indices]
    X_val_numerical = X_val[:, [i for i in range(X_val.shape[1]) if i not in categorical_indices]].astype(float)
    X_val_cat_imp = imputer.transform(X_val_categorical)
    X_val_cat_imp = X_val_cat_imp.astype(str)

    X_val_cat_enc = transform_with_default(ordinal_encoder, X_val_cat_imp)

    X_val_encoded = np.hstack([X_val_numerical, X_val_cat_enc])

    return X_train_encoded, X_val_encoded
