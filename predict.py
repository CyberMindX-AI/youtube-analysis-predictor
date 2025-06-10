import re
import joblib
import googleapiclient.discovery
import requests
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
load_dotenv()  

# Load trained models
model_likes = joblib.load("like_count_predictor_model.pkl")
model_views = joblib.load("view_count_predictor_model.pkl")

# Setup YouTube API
YOUTUBE_API_KEY= os.getenv("YOUTUBE_API_KEY")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# === Extract video ID from URL ===

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_video_data(video_id):
    """Fetch video data from YouTube API"""
    if not YOUTUBE_API_KEY:
        print("❌ YOUTUBE_API_KEY not found in environment variables")
        return None
    
    try:
        # YouTube Data API endpoint
        url = f"https://www.googleapis.com/youtube/v3/videos"
        params = {
            'part': 'snippet,statistics,contentDetails',
            'id': video_id,
            'key': YOUTUBE_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('items'):
            print(f"❌ No video found with ID: {video_id}")
            return None
        
        video = data['items'][0]
        snippet = video['snippet']
        statistics = video['statistics']
        content_details = video['contentDetails']
        
        # Calculate days since published
        published_at = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
        days_since_published = (datetime.now(published_at.tzinfo) - published_at).days
        
        video_data = {
            'title': snippet.get('title', ''),
            'description': snippet.get('description', ''),
            'tags': snippet.get('tags', []),
            'categoryId': snippet.get('categoryId', ''),
            'publishedAt': snippet.get('publishedAt', ''),
            'duration': content_details.get('duration', 'PT0S'),
            'viewCount': int(statistics.get('viewCount', 0)),
            'likeCount': int(statistics.get('likeCount', 0)),
            'commentCount': int(statistics.get('commentCount', 0)),
            'thumbnail': snippet['thumbnails']['high']['url'],
            'days': days_since_published
        }
        
        return video_data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error fetching video data: {e}")
        return None

# === Convert ISO 8601 duration to seconds ===
def convert_duration_to_seconds(duration):
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    hours = int(match.group(1)) if match and match.group(1) else 0
    minutes = int(match.group(2)) if match and match.group(2) else 0
    seconds = int(match.group(3)) if match and match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds
# === Predict views and likes ===
def predict_views_and_likes(video_data):
    duration_seconds = convert_duration_to_seconds(video_data['duration'])

    # IMPORTANT: Use only numerical/categorical features the model was trained on
    # Here, we're using:
    # categoryId, duration_seconds, likeCount (or predicted views), commentCount, days
    view_features = [[
        video_data['categoryId'],
        duration_seconds,
        video_data['likeCount'],        # Use actual like count here for views prediction
        video_data['commentCount'],
        video_data['days']
    ]]
    # Predict views (log-transformed during training, so use expm1 to reverse)
    predicted_views = int(np.expm1(model_views.predict(view_features)[0]))

    like_features = [[
        video_data['categoryId'],
        duration_seconds,
        predicted_views,                # Use predicted views for likes prediction
        video_data['commentCount'],
        video_data['days']
    ]]
    predicted_likes = int(np.expm1(model_likes.predict(like_features)[0]))

    return predicted_views, predicted_likes

# === Estimate how long it’ll take to reach predicted views/likes ===
def estimate_days_to_reach(video_data, predicted_views, predicted_likes):
    current_views = video_data['viewCount']
    current_likes = video_data['likeCount']
    days = video_data['days']

    # Prevent division by zero
    view_rate = current_views / days if days else 1
    like_rate = current_likes / days if days else 1

    days_to_views = max(0, (predicted_views - current_views) / view_rate) if view_rate > 0 else float('inf')
    days_to_likes = max(0, (predicted_likes - current_likes) / like_rate) if like_rate > 0 else float('inf')

    return round(days_to_views), round(days_to_likes)

# === Display the output clearly ===
def show_comparison(video_data, predicted_views, predicted_likes, days_to_views, days_to_likes):
    print(" Title:", video_data['title'])
    print(" Thumbnail:", video_data['thumbnail'])
    print(" Video Link:", video_data['video_url'])

    print("\n Current Stats:")
    print("    Views:", video_data['viewCount'])
    print("    Likes:", video_data['likeCount'])

    print("\n Predicted Stats:")
    print("    Predicted Views:", predicted_views)
    print("    Predicted Likes:", predicted_likes)

    print("\n Estimated Days to Reach Predicted Stats:")
    print("    Days to Views:", days_to_views)
    print("    Days to Likes:", days_to_likes)
