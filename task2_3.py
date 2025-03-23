import json
import time
from pyspark import SparkContext
import os
import sys
import numpy as np

#here's the plan: use item cf predictions as a feature in the model

# along the way: get the item based cf better
# : hone in on important features for the normal regression model
#: do a little bit of finetuning on learning rate and maybe some regularization techniques
# : maybe some kind of error correction model



#os.environ['PYSPARK_Python'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

train_path = sys.argv[1]
val = sys.argv[2]
output = sys.argv[3]

#input_file = sys.argv[1]
#val_file = sys.argv[2]
#output = sys.argv[3]

start = time.time()
sc = SparkContext('local[*]', 'task21')
input_file = train_path +"\\yelp_train.csv"
review_rdd = sc.textFile(input_file)
rdd = review_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
parsed_rdd = rdd.map(lambda x: x.split(','))


business_data = parsed_rdd.map(lambda x: (x[1], x[0])).mapValues(lambda v: [v]).reduceByKey(lambda a, b: a + b).mapValues(set).cache()

users_data = parsed_rdd.map(lambda x: (x[0], x[1])).mapValues(lambda v: [v]).reduceByKey(lambda a, b: a + b).mapValues(set).cache()

ratings_data = parsed_rdd.map(lambda x: ((x[1], x[0]), float(x[2]))).cache()

user_ratings = ratings_data.map(lambda x: (x[0][1], x[1]))  # (user_id, rating)


user_rating_sum_count = user_ratings.combineByKey(
    lambda rating: (rating, 1),  # Create a (sum, count) tuple for the first rating
    lambda acc, rating: (acc[0] + rating, acc[1] + 1),  # Update the (sum, count) tuple
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # Combine the tuples from different partitions
)

user_averages = user_rating_sum_count.mapValues(lambda x: x[0] / x[1])  

user_averages_dict = user_averages.collectAsMap()

business_ratings = ratings_data.map(lambda x: (x[0][0], x[1]))  # (business_id, rating)

# Step 2: Aggregate ratings by business_id (sum the ratings and count the number of ratings)
business_rating_sum_count = business_ratings.combineByKey(
    lambda rating: (rating, 1),  # Create a (sum, count) tuple for the first rating
    lambda acc, rating: (acc[0] + rating, acc[1] + 1),  # Update the (sum, count) tuple
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])  # Combine the tuples from different partitions
)

# Step 3: Calculate the average rating for each business
business_averages = business_rating_sum_count.mapValues(lambda x: x[0] / x[1])

# Step 4: Collect the result as a dictionary
business_avg_dict = business_averages.collectAsMap()

ratings_dict = ratings_data.collectAsMap()  

users_data_dict = users_data.collectAsMap()

business_data_dict = business_data.collectAsMap()

def calculate_similarity(business1, business2, shrinkage=10):
   
    # Find co-rated users for both businesses
    if business1 in business_data_dict and business2 in business_data_dict:
        co_rated_users = business_data_dict[business1] & business_data_dict[business2]
    else:
        return 0.0
    
    num_co_raters = len(co_rated_users)
    
    if num_co_raters == 0:
        return 0.0  # No co-rated users, similarity is zero
    
    # Ratings for bus1 and bus2 from co-rated users
    ratings_bus1 = [float(ratings_dict.get((business1, user), 0)) for user in co_rated_users]
    ratings_bus2 = [float(ratings_dict.get((business2, user), 0)) for user in co_rated_users]
    
    # Calculate numerator and denominators for cosine similarity
    numerator = sum([x * y for x, y in zip(ratings_bus1, ratings_bus2)])
    denominator = (sum([x ** 2 for x in ratings_bus1]) ** 0.5) * (sum([y ** 2 for y in ratings_bus2]) ** 0.5)
    
    if denominator == 0:
        return 0.0  # Avoid division by zero
    
    # Cosine similarity
    raw_similarity = numerator / denominator
    
    # Shrinkage adjustment: Incorporate the number of co-raters
    adjusted_similarity = (num_co_raters * raw_similarity) / (num_co_raters + shrinkage)
    
    return adjusted_similarity



#calculate_similarity('4JNXUYY8wbaaDmk3BPzlWw','FaHADZARwnY4yvlvpnsfGA')

#business_data_dict['UkwuYrzwLXI_QidhVzUkgg']

business_review_count = parsed_rdd.map(lambda x: (x[1], 1)) \
                                  .reduceByKey(lambda a, b: a + b)

# Step 2: Get the top 20 most popular businesses
top_20_businesses = business_review_count.sortBy(lambda x: -x[1]).take(20)

# Extract only the business IDs
top_20_business_ids = [business_id for business_id, count in top_20_businesses]

top_20_business_pairs = [(top_20_business_ids[i], top_20_business_ids[j]) 
                         for i in range(len(top_20_business_ids)) 
                         for j in range(i + 1, len(top_20_business_ids))]

# Dictionary to store the precomputed similarities
precomputed_similarities = {}

# Step 4: Precompute similarities for all pairs of the top 20 businesses
for bus1, bus2 in top_20_business_pairs:
    similarity = calculate_similarity(bus1, bus2)
    similarity_key = (bus1, bus2) if bus1 < bus2 else (bus2, bus1)
    precomputed_similarities[similarity_key] = similarity

# Collect and store the precomputed similarities in a dictionary
#precomputed_similarities 

#precomputed_similarities = {}
def calculate_rating_with_features(user_id, target_business):
    # Get the businesses the user has visited and rated
    visited_businesses = users_data_dict.get(user_id, set())
    # If the user has not visited any businesses, return default values
    if not visited_businesses:
        user_avg_rating = user_averages_dict.get(user_id, 3.45)
        business_avg_rating = business_avg_dict.get(target_business, 3.45)
        predicted_rating = (user_avg_rating + business_avg_rating) / 2
        return predicted_rating
    # Calculate similarities between the target business and the businesses the user has rated
    similarities = []
    for bus1 in visited_businesses:
        similarity_key = (bus1, target_business) if bus1 < target_business else (target_business, bus1)
        similarity = precomputed_similarities.get(similarity_key, None)
        if similarity is None:
            similarity = calculate_similarity(bus1, target_business)
            if similarity > 0:
                precomputed_similarities[similarity_key] = similarity
        if similarity and similarity > 0:
            rating = ratings_dict.get((bus1, user_id), 0)
            similarities.append((similarity, rating))
    # If no similarities found, return average of user and business averages
    if not similarities:
        user_avg_rating = user_averages_dict.get(user_id, 3.45)
        business_avg_rating = business_avg_dict.get(target_business, 3.45)
        predicted_rating = (user_avg_rating + business_avg_rating) / 2
        return predicted_rating
    # Get the top N similar businesses
    top_N = 15  # You can adjust this number
    top_similarities = sorted(similarities, key=lambda x: -x[0])[:top_N]
    # Calculate the weighted sum of ratings and total similarity weights
    weighted_sum = sum([sim * rating for sim, rating in top_similarities])
    total_weights = sum([abs(sim) for sim, _ in top_similarities])
    # Default rating if no valid weights are found
    if total_weights == 0:
        user_avg_rating = user_averages_dict.get(user_id, 3.45)
        business_avg_rating = business_avg_dict.get(target_business, 3.45)
        predicted_rating = (user_avg_rating + business_avg_rating) / 2
    else:
        predicted_rating = weighted_sum / total_weights
    # Ensure the predicted rating is within the valid range
    predicted_rating = max(min(predicted_rating, 5.0), 1.0)
    return predicted_rating


# Example usage:\
#val_file = train_path + val

val_rdd= sc.textFile(val)
rdd = val_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
finalval_rdd = rdd.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))

result_rdd = finalval_rdd.map(lambda pair: (pair, calculate_rating_with_features(pair[0], pair[1]))).collectAsMap()

train_result_rdd = parsed_rdd.map(lambda x: (x[0], x[1])).map(lambda pair: (pair, calculate_rating_with_features(pair[0], pair[1]))).collectAsMap()



#####
#####
#Model
#####
#####

import json
import time
from pyspark import SparkContext
import os
import sys
from xgboost import XGBRegressor
import numpy as np

#os.environ['PYSPARK_Python'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


#sc = SparkContext('local[*]', 'task22')
#start = time.time()

#train_path = "C:\\DSCI_553\\HW3"

business_info_file = train_path +"\\business.json"
business_info_rdd = sc.textFile(business_info_file).map(lambda x: json.loads(x))

def extract_business_features(business):
    if business is None:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None)
    attributes = business.get('attributes', {}) if business.get('attributes') else {}
    #categories = business.get('categories', "") if business.get('categories') else "
    return (
        business.get('business_id'),                      
        business.get('stars', None),                          
        business.get('review_count', None),                   
#       attributes.get('RestaurantsPriceRange2', None),    
        business.get('is_open', None),                       
    )


final_bus_info = business_info_rdd.map(extract_business_features)
#x = final_bus_info.collect()



user_info_file = train_path +"\\user.json"
user_info_rdd = sc.textFile(user_info_file).map(lambda x: json.loads(x))

def extract_user_features(user):
    if user is None:
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
    
    return (
        user.get('user_id', None),                    
        #user.get('name', None),                        
        user.get('review_count', None),             
        #user.get('yelping_since', None),               
        #user.get('friends', "None"),                   
        user.get('useful', None),                         
        user.get('funny', None),                           
        user.get('cool', None),                            
        user.get('fans', None),                       
        #user.get('elite', "None"),                     
        user.get('average_stars', None),                 

    )

final_user_info = user_info_rdd.map(extract_user_features)





checkin_info_file = train_path +"\\checkin.json"
checkin_info_rdd = sc.textFile(checkin_info_file).map(lambda x: json.loads(x))


def extract_checkin_features(business):
    time_data = business.get("time", {})
    # Initialize variables to store aggregated features
    total_interactions = 0
    interactions_by_day = {}
    interactions_by_hour = {}
    # To track the most active day and hour
    most_active_day = None
    most_active_hour = None
    max_day_interactions = 0
    max_hour_interactions = 0
    # Track the maximum number of interactions overall for the business
    max_interactions_overall = 0
    # Iterate over the "time" field to aggregate data
    for key, value in time_data.items():
        # Split the key into day and hour
        day, hour = key.split("-")
        hour = int(hour)
        # Update total interactions
        total_interactions += value
        # Update interactions by day
        if day in interactions_by_day:
            interactions_by_day[day] += value
        else:
            interactions_by_day[day] = value
        # Update interactions by hour
        if hour in interactions_by_hour:
            interactions_by_hour[hour] += value
        else:
            interactions_by_hour[hour] = value
        # Track the most active day
        if interactions_by_day[day] > max_day_interactions:
            max_day_interactions = interactions_by_day[day]
            most_active_day = day
        # Track the most active hour
        if interactions_by_hour[hour] > max_hour_interactions:
            max_hour_interactions = interactions_by_hour[hour]
            most_active_hour = hour
        # Track the maximum number of interactions for any given time slot
        if value > max_interactions_overall:
            max_interactions_overall = value
    
    return (
        business.get("business_id"),
        total_interactions,
        #interactions_by_day,   
        #interactions_by_hour,  
        #most_active_day,
        #most_active_hour,
        #max_interactions_overall  #max interactions overall for a business
    )


final_checkin_info = checkin_info_rdd.map(extract_checkin_features)
#final_checkin_info.take(2)


final_user_features_dict = final_user_info.map(lambda x: (x[0], x[1:])).collectAsMap()

business_features_rdd = final_bus_info.map(lambda x: (x[0], x[1:]))

checkin_features_rdd = final_checkin_info.map(lambda x: (x[0], x[1:]))
combined_business_dict = business_features_rdd.join(checkin_features_rdd).collectAsMap()

user_features_broadcast = sc.broadcast(final_user_features_dict)
business_features_broadcast = sc.broadcast(combined_business_dict)

input_file = train_path + "\\yelp_train.csv"

#review_rdd = sc.textFile(train + '/yelp_train.csv')

review_rdd = sc.textFile(input_file)
rdd = review_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])  
parsed_rdd = rdd.map(lambda x: x.split(','))  
parsed_rdd = parsed_rdd.map(lambda x: (x[0], x[1], float(x[2])))  # (user_id, business_id, stars)

def get_combined_features(record):
    user_id, business_id, stars = record
    user_features = user_features_broadcast.value.get(user_id, [None, None,None,None,None,None])
    business_features = business_features_broadcast.value.get(business_id, [None, None, None, None])
    predicted_rating = train_result_rdd.get((user_id, business_id), None)  # Default value for predicted rating
    
    combined_features = list(user_features) + list(business_features) + [predicted_rating]
    return (user_id, business_id, stars, combined_features)

#next(iter(combined_business_dict.items()))
def flatten_features(features):
    flat_list = []
    for item in features:
        if isinstance(item, tuple):  # If the item is a tuple, flatten it
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list


# Apply the function to your parsed RDD
final_train_rdd = parsed_rdd.map(get_combined_features)

user_business_rdd = final_train_rdd.map(lambda x: (x[0], x[1]))  # (user_id, business_id)
X_y_rdd = final_train_rdd.map(lambda x: (flatten_features(x[3]), x[2]))  # (flattened features, target)

X_rdd = X_y_rdd.map(lambda x: x[0])  # Flattened Features (X)
y_rdd = X_y_rdd.map(lambda x: x[1])  # Target (y)



user_business = np.array(user_business_rdd.collect())
X = np.array(X_rdd.collect())  
y = np.array(y_rdd.collect())

#np.save('X.npy', X)
#np.save('y.npy', y)
"""
val = train_path + "\\yelp_val.csv"
val_rdd = sc.textFile(val)
val_rdd = val_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
parsed_val_rdd = val_rdd.map(lambda x: x.split(','))


def get_combined_features_val(record):
    user_id, business_id, stars = record
    user_features = user_features_broadcast.value.get(user_id, [None, None,None,None,None,None])
    business_features = business_features_broadcast.value.get(business_id, [None, None, None, None])
    predicted_rating = train_result_rdd.get((user_id, business_id), None)  # Default value for predicted rating
    
    combined_features = list(user_features) + list(business_features) + [predicted_rating]
    return (user_id, business_id, stars, combined_features)

# Apply the function to the validation RDD
final_val_rdd = parsed_val_rdd.map(get_combined_features_val)

# Extract X and y from the validation set
val_user_business_rdd = final_val_rdd.map(lambda x: (x[0], x[1]))  # (user_id, business_id)
X_val_yval_rdd = final_val_rdd.map(lambda x: (flatten_features(x[3]), x[2]))  # (flattened features, stars)

X_val_rdd = X_val_yval_rdd.map(lambda x: x[0])  # Flattened features (X)
y_val_rdd = X_val_yval_rdd.map(lambda x: x[1])  # Target (y)

# Collect X_val and y_val for local use
val_user_business = np.array(val_user_business_rdd.collect())
X_val = np.array(X_val_rdd.collect())
y_val = np.array(y_val_rdd.collect())
"""
#np.save('X_val.npy', X_val)
#np.save('y_val.npy', y_val)


val_rdd = sc.textFile(val)
val_rdd = val_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
parsed_val_rdd = val_rdd.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))  # Only (user_id, business_id)

def get_combined_features_val(record):
    user_id, business_id = record
    user_features = user_features_broadcast.value.get(user_id, [None, None, None, None, None, None])
    business_features = business_features_broadcast.value.get(business_id, [None, None, None, None])
    predicted_rating = train_result_rdd.get((user_id, business_id), None)  # Default value for predicted rating
    combined_features = list(user_features) + list(business_features) + [predicted_rating]
    return (user_id, business_id, combined_features)

# Apply the function to the validation RDD
final_val_rdd = parsed_val_rdd.map(get_combined_features_val)


val_user_business_rdd = final_val_rdd.map(lambda x: (x[0], x[1]))  # (user_id, business_id)
X_val_rdd = final_val_rdd.map(lambda x: flatten_features(x[2]))  # Flattened features (X)


val_user_business = np.array(val_user_business_rdd.collect())
X_val = np.array(X_val_rdd.collect())



model = XGBRegressor(
    objective='reg:squarederror',  
    eval_metric='rmse',
    #reg_lambda=9.0,                  
    #reg_alpha=0.2765119705933928,             
    max_depth=14,                   
    learning_rate=0.061,            
    subsample=0.8,                
    colsample_bytree=0.6,          
    min_child_weight=100,            
    #gamma=0,                       
    n_estimators=300               
)



model.fit(X, y)

preds = model.predict(X_val)

#output= "output_23.csv"

with open(output, 'w') as f_out:
    f_out.write("user_id,business_id,prediction\n")
    
    for u,p in zip(val_user_business,preds):
        f_out.write(f"{u[0]},{u[1]},{p}\n")

end = time.time()

print(end - start)


#dev
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# Step 1: Read both CSVs
predictions_df = pd.read_csv('C:\\DSCI_553\\output_23.csv')  # Contains predicted values
actuals_df = pd.read_csv("C:\\DSCI_553\\HW3\\yelp_val.csv")  # Contains actual values

# Step 2: Merge the two dataframes on 'user_id' and 'business_id'
merged_df = pd.merge(predictions_df, actuals_df, on=['user_id', 'business_id'])

rmse = mean_squared_error(merged_df['stars'], merged_df['prediction'], squared=False) 


# Step 4: Print the RMSE
print(f"RMSE over the entire dataset: {rmse}")




sc.stop()

#####
#####
#combining together
#####
#####


train_model_dict = { (u[0], u[1]): p for u, p in zip(user_business, classtrainpreds) }
test_model_dict = { (u[0], u[1]): p for u, p in zip(val_user_business, preds) }

actual_rdd = parsed_rdd.map(lambda x: ((x[0], x[1]), float(x[2])))
actual_dict = actual_rdd.collectAsMap()           


#result_rdd 
#train_result_rdd 

train_combined_dict = {
    key: (train_model_dict.get(key), train_result_rdd.get(key), actual_dict.get(key))
    for key in set(train_model_dict)  | set(train_result_rdd) | set(actual_dict) # Union of all keys
}

test_combined_dict = {
    key: (test_model_dict.get(key), result_rdd.get(key))
    for key in set(test_model_dict) | set(result_rdd)  # Union of keys from both dictionaries
}


def assign_class(preds):
    pred1, pred2, actual = preds
    # Ensure None values don't cause issues
    if pred1 is None or pred2 is None or actual is None:
        return (pred1, pred2, actual, None)  # Cannot assign class, missing values
    # Assign class based on which prediction is closer to the actual value
    class_label = 0 if abs(pred1 - actual) < abs(pred2 - actual) else 1
    return (pred1, pred2, actual, class_label)

# Apply the class assignment to the train_combined_dict
train_class_dict = {
    key: assign_class(preds)
    for key, preds in train_combined_dict.items()
}

# Assuming `train_class_dict` has the structure:
# { (user_id, business_id): (pred1, pred2, actual, class_label) }

# Function to get the combined features for a given user-business pair
def get_combined_features_for_dict(user_id, business_id):
    # Get user and business features
    user_features = user_features_broadcast.value.get(user_id, [3.45, 0, 0, 0, 3])
    business_features = business_features_broadcast.value.get(business_id, [3.45, 0, 0, 0, 0])
    combined_features = list(user_features) + list(business_features)
    return flatten_features(combined_features)

# Flatten features (helper function)
def flatten_features(features):
    flat_list = []
    for item in features:
        if isinstance(item, tuple):
            flat_list.extend(item)  # Flatten any tuple
        else:
            flat_list.append(item)
    return flat_list

# Construct X (features) and y (class label) using train_class_dict
X_y_combined = [
    (get_combined_features_for_dict(user_id, business_id), class_info[3])  # class_info[3] is the class label
    for (user_id, business_id), class_info in train_class_dict.items()
    if class_info[3] is not None  # Exclude pairs with no class label
]

# Split into X and y
X = [item[0] for item in X_y_combined]  # Features (X)
y = [item[1] for item in X_y_combined]  # Class labels (y)


X = np.array(X)
y = np.array(y)

from xgboost import XGBClassifier

class_model = XGBClassifier(
    objective='binary:logistic',  
    eval_metric='logloss',
    reg_lambda=9.0,                  
    #reg_alpha=0.2765119705933928,             
    max_depth=16,                   
    learning_rate=0.051,            
    subsample=0.8,                
    colsample_bytree=0.6,          
    min_child_weight=100,            
    gamma=0,                       
    n_estimators=300               
)

class_model.fit(X,y)

class_preds = class_model.predict(X_val)

# Create the dictionary correctly using a dictionary comprehension
test_class_model_data = dict([((u[0], u[1]), p) for u, p in zip(val_user_business, class_preds)])


# Function to select the correct prediction based on the class_pred (from `test_class_model_data`)
def select_prediction_from_dict(test_class_model_data, test_combined_dict):
    selected_predictions = {
        key: (test_combined_dict[key][0] if class_pred == 1 else test_combined_dict[key][1])
        for key, class_pred in test_class_model_data.items()
        if key in test_combined_dict  # Ensure the key exists in the test_combined_dict
    }
    return selected_predictions

# Apply the function to get the selected predictions
final_selected_predictions = select_prediction_from_dict(test_class_model_data, test_combined_dict)


output_file = "output_23.csv"  # Define your output file path

with open(output_file, 'w') as f_out:
    # Write header
    f_out.write("user_id,business_id,prediction\n")
    
    # Iterate over the selected predictions and write to the file
    for (user_id, business_id), prediction in final_selected_predictions.items():
        f_out.write(f"{user_id},{business_id},{prediction}\n")


end = time.time()

print(end - start)

"""

