import numpy as np 
import re
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

# === 1. Load dataset ===
dataset = pd.read_csv('videos_data.csv')
dataset.fillna(0, inplace=True)

# === 2. Convert ISO 8601 duration to seconds ===
def convert_duration_to_seconds(duration):
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    hours = int(match.group(1)) if match and match.group(1) else 0
    minutes = int(match.group(2)) if match and match.group(2) else 0
    seconds = int(match.group(3)) if match and match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

dataset['duration_sec'] = dataset['duration'].apply(convert_duration_to_seconds)
dataset['categoryId'] = pd.to_numeric(dataset['categoryId'], errors='coerce').fillna(0).astype(int)


dataset['days'] = 7  # Static value for now (predicting view/like counts after 7 days)

# === 4. View Count Model ===
features = ['categoryId', 'duration_sec', 'likeCount', 'commentCount', 'days']
target_views = 'viewCount'
X_views = dataset[features]
y_views = np.log1p(dataset[target_views])  # log-transform

X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X_views, y_views, test_size=0.2, random_state=42)

model_views = DecisionTreeRegressor(random_state=0)
model_views.fit(X_train_v, y_train_v)
y_pred_v = model_views.predict(X_test_v)

# Evaluate with inverse transform
y_test_v_exp = np.expm1(y_test_v)
y_pred_v_exp = np.expm1(y_pred_v)

print("\n📈 VIEW COUNT MODEL:")
print("Mean Squared Error (log scale):", mean_squared_error(y_test_v, y_pred_v))
print("Mean Absolute Error (original scale):", mean_absolute_error(y_test_v_exp, y_pred_v_exp))
print("R2 Score (log scale):", r2_score(y_test_v, y_pred_v))

r2_v = r2_score(y_test_v, y_pred_v)
if r2_v > 0.8:
    print("✅ The view count model is good.")
elif r2_v > 0.5:
    print("⚠️ The view count model is decent but can be improved.")
else:
    print("❌ The view count model is poor. Improve data or features.")

# Save view model
joblib.dump(model_views, 'view_count_predictor_model.pkl')
print("💾 View count model saved as 'view_count_predictor_model.pkl'.")

# === 5. Like Count Model ===
# Add predicted viewCount to dataset as a feature
dataset['predicted_viewCount'] = np.expm1(model_views.predict(X_views))

features_likes = ['categoryId', 'duration_sec', 'predicted_viewCount', 'commentCount', 'days']
target_likes = 'likeCount'
X_likes = dataset[features_likes]
y_likes = np.log1p(dataset[target_likes])

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_likes, y_likes, test_size=0.2, random_state=42)

model_likes = DecisionTreeRegressor(random_state=0)
model_likes.fit(X_train_l, y_train_l)
y_pred_l = model_likes.predict(X_test_l)

# Evaluate with inverse transform
y_test_l_exp = np.expm1(y_test_l)
y_pred_l_exp = np.expm1(y_pred_l)

print("\n👍 LIKE COUNT MODEL:")
print("Mean Squared Error (log scale):", mean_squared_error(y_test_l, y_pred_l))
print("Mean Absolute Error (original scale):", mean_absolute_error(y_test_l_exp, y_pred_l_exp))
print("R2 Score (log scale):", r2_score(y_test_l, y_pred_l))

r2_l = r2_score(y_test_l, y_pred_l)
if r2_l > 0.8:
    print("✅ The like count model is good.")
elif r2_l > 0.5:
    print("⚠️ The like count model is decent but can be improved.")
else:
    print("❌ The like count model is poor. Improve data or features.")

# Save like model
joblib.dump(model_likes, 'like_count_predictor_model.pkl')
print("💾 Like count model saved as 'like_count_predictor_model.pkl'.")
