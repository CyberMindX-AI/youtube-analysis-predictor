# Import necessary libraries
import re
from urllib.parse import urlparse, parse_qs
import googleapiclient.discovery
import requests
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
load_dotenv()  

# Set your YouTube API key
apikey = os.getenv("YOUTUBE_API_KEY")
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=apikey)

# Search YouTube videos using a query
def search_video_url(query, youtube, max_results=50):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    urls = []
    for item in response['items']:
        video_id = item['id']['videoId']
        url = f"https://www.youtube.com/watch?v={video_id}"
        urls.append(url)
    return urls

# Fetch video metadata using the video ID
def fetch_video_data(video_id):
    api_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": apikey
    }
    response = requests.get(api_url, params=params)
    
    if response.status_code != 200:
        print(f'Error fetching data for video ID {video_id}: {response.status_code}')
        return None

    response_data = response.json()
    if 'items' not in response_data or not response_data['items']:
        print(f'No data found for video ID {video_id}')
        return None

    video_data = response_data["items"][0]
    snippet = video_data.get("snippet", {})
    statistics = video_data.get("statistics", {})
    contentDetails = video_data.get("contentDetails", {})

    published_at = snippet.get("publishedAt", "")
    days = None
    if published_at:
        published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = (now - published_dt).days

    results = {
        "title": snippet.get("title", ""),
        "description": snippet.get("description", "").replace('\n', ' ').replace('\r', ' '),
        "tags": ','.join(snippet.get("tags", [])),
        "publishedAt": published_at,
        "categoryId": snippet.get("categoryId", ""),
        "viewCount": int(statistics.get("viewCount", 0)),
        "likeCount": int(statistics.get("likeCount", 0)),
        "commentCount": int(statistics.get("commentCount", 0)),
        "duration": contentDetails.get("duration", ""),
        "thumbnail": snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
        "days": days
    }
    return results

# Save metadata of all videos to a CSV file
def save_videos_to_csv(videos_data, filename="videos_data.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'title', 'description', 'tags', 'publishedAt', 'categoryId',
            'viewCount', 'likeCount', 'commentCount', 'duration', 'thumbnail','days'
        ])
        for video in videos_data:
            writer.writerow([
                video['title'],
                video['description'],
                video['tags'],
                video['publishedAt'],
                video['categoryId'],
                video['viewCount'],
                video['likeCount'],
                video['commentCount'],
                video['duration'],
                video['thumbnail'],
                video['days']
            ])
    print(f"✅ All video data saved to {filename}")

# Main function to handle input and loop through queries
def main():
    search_terms = "ai tools 2025, chatgpt tutorials, midjourney ai prompts, gta 6 leaks gameplay, roblox funny moments, minecraft hardcore challenge, make money online 2025, crypto market news, dropshipping success stories, viral tiktok compilation, funny youtube shorts, pranks gone wrong 2025, creepy ai horror story, unsolved mystery short, study hacks for students"

    queries = [term.strip() for term in search_terms.split(',')]
    all_video_data = []

    for query in queries:
        print(f"\n🔍 Searching videos for: {query}")
        urls = search_video_url(query, youtube, max_results=20)
        print(f"📺 Found {len(urls)} videos for '{query}'")

        for url in urls:
            try:
                video_id = parse_qs(urlparse(url).query).get("v")[0]
                data = fetch_video_data(video_id)
                if data:
                    all_video_data.append(data)
            except Exception as e:
                print(f"⚠️ Error processing video: {e}")

    if all_video_data:
        save_videos_to_csv(all_video_data)
    print("\n Video data collection complete.")

# Run the script
if __name__ == "__main__":
    main()
