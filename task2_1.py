import json
import time
from pyspark import SparkContext
import os
import sys


os.environ['PYSPARK_Python'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

input_file = sys.argv[1]
val_file = sys.argv[2]
output = sys.argv[3]

start = time.time()
sc = SparkContext('local[*]', 'task21')
#input_file = "C:\\DSCI_553\\HW3\\yelp_train.csv"
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


def calculate_similarity(business1, business2):
    # Find co-rated users for both businesses
    if business1 in business_data_dict and business2 in business_data_dict:
        # Use set intersection to find co-rated users
        co_rated_users = business_data_dict[business1] & business_data_dict[business2]
    else:
        return 0.0
    #print(f"Co-rated users for {business1} and {business2}: {co_rated_users}")
    if len(co_rated_users) <= 2:
        # Get the average ratings for both businesses
        avg_rating_bus1 = business_avg_dict.get(business1, 0)
        avg_rating_bus2 = business_avg_dict.get(business2, 0)
        
        # Calculate similarity based on the absolute difference between average ratings
        max_rating = 5.0  # Assuming ratings are on a 5-point scale
        similarity = 1 - (abs(avg_rating_bus1 - avg_rating_bus2) / max_rating)
        
        # Ensure similarity is between 0 and 1
        return max(similarity, 0)
    #print("users: " + str(len(co_rated_users)))
    # Ratings for bus1 and bus2 from co-rated users
    ratings_bus1 = [float(ratings_dict.get((business1, user), 0)) for user in co_rated_users]
    ratings_bus2 = [float(ratings_dict.get((business2, user), 0)) for user in co_rated_users]
    #print(ratings_bus1, ratings_bus2)
    # Compute averages
    avg_bus1 = sum(ratings_bus1) / len(ratings_bus1)
    avg_bus2 = sum(ratings_bus2) / len(ratings_bus2)
    # Normalize the ratings by subtracting the average rating
    norm_ratings_bus1 = [x - avg_bus1 for x in ratings_bus1]
    norm_ratings_bus2 = [x - avg_bus2 for x in ratings_bus2]
    # Calculate the covariance (numerator of Pearson correlation)
    covariance = sum([x * y for x, y in zip(norm_ratings_bus1, norm_ratings_bus2)])
    # Calculate the product of standard deviations (denominator of Pearson correlation)
    std_bus1 = (sum([x ** 2 for x in norm_ratings_bus1]) ** 0.5)
    std_bus2 = (sum([x ** 2 for x in norm_ratings_bus2]) ** 0.5)
    # If either standard deviation is zero, return zero similarity
    if std_bus1 == 0 or std_bus2 == 0:
        return 0.0
    # Calculate Pearson correlation (similarity)
    similarity = covariance / (std_bus1 * std_bus2)
    #print(similarity)
    return similarity

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

def calculate_rating(user_id, target_business):
    # Get the businesses the user has visited and rated
    visited_businesses = users_data_dict.get(user_id, [])
    # Calculate similarities between the target business and the businesses the user has rated
    weights = []
    for bus1 in visited_businesses:
        similarity_key = (bus1, target_business) if bus1 < target_business else (target_business, bus1)
        if similarity_key in precomputed_similarities:
            #print("pre")
            similarity = precomputed_similarities[similarity_key]
        else:
            # If not precomputed, calculate it
            similarity = calculate_similarity(bus1, target_business)
            # Store the calculated similarity
            precomputed_similarities[similarity_key] = similarity
        if abs(similarity) > 0:
            # Fetch the user's rating for bus1 from the ratings_dict
            rating = ratings_dict.get((bus1, user_id), 0)
            weights.append((similarity, rating))
    # Get the top 10 similar businesses
    top_weights = sorted(weights, key=lambda x: -x[0])[:10]
    #print(top_weights)
    # Calculate the weighted sum of ratings
    #top_weights = [(0.5000000000000001, 5.0), (-0.8528028654224417, 4.0)]
    weighted_sum = sum([sim * rating for sim, rating in top_weights])
    total_weights = sum([abs(sim) for sim, _ in top_weights])
    # If no weights or all weights are zero, return a default rating
    if total_weights == 0:
        if user_id in user_averages_dict:
            return user_averages_dict[user_id]
        return 3.45  # Default rating
    # Return the weighted average
    return (user_averages_dict[user_id] + (weighted_sum / total_weights))/2

#top_weights = [(0.5000000000000001, 5.0), (-0.8528028654224417, 4.0)]
#calculate_rating('I-4KVZ9lqHhk8469X9FvhA','qE9yIXn2GQb2-4a_qOOKzg')

#val_file = "C:\\DSCI_553\\HW3\\yelp_val.csv"

val_rdd= sc.textFile(val_file)
rdd = val_rdd.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])

#(userid, businessid)
finalval_rdd = rdd.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1]))


result_rdd = finalval_rdd.map(lambda pair: (pair, calculate_rating(pair[0], pair[1])))

#test_rdd = rdd.map(lambda x: x.split(',')).map(lambda x: x[2])

#from math import sqrt
#predicted_rdd = result_rdd.map(lambda x: x[1])  # Only take the predicted ratings


#actual_rdd = test_rdd.map(lambda x: float(x))  # Actual ratings from the test data

# Step 3: Zip the two RDDs together (align predicted and actual ratings)
#predicted_and_actual_rdd = predicted_rdd.zip(actual_rdd)  # (predicted, actual)

# Step 4: Compute squared errors
#squared_errors_rdd = predicted_and_actual_rdd.map(lambda x: (x[0] - x[1]) ** 2)

# Step 5: Calculate mean squared error (MSE)
#mse = squared_errors_rdd.mean()

# Step 6: Compute the RMSE (Root Mean Squared Error)
#rmse = sqrt(mse)

# Print or return the RMSE
#print(f"RMSE: {rmse}")

csv_rdd = result_rdd.map(lambda x: f"{x[0][0]},{x[0][1]},{x[1]}").collect()


#output_file = "predictions_with_header21.csv"

# Open the final output file and write the header
with open(output, 'w') as f_out:
    f_out.write("user_id,business_id,prediction\n")
    
    for line in csv_rdd:
        f_out.write(line + "\n")

end = time.time()

print(end - start)

#get user and all their businesses they've reviewed
#get business and all users who have reviewed them

