import streamlit as st
import re
import joblib
import google.generativeai as genai
from predict import (
    get_video_id,
    fetch_video_data,
    predict_views_and_likes,
    estimate_days_to_reach,
    convert_duration_to_seconds
)
from dotenv import load_dotenv
import os

load_dotenv()  

gemini_api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('models/gemini-2.0-flash-lite-preview')

# Load ML models
model_likes = joblib.load("like_count_predictor_model.pkl")
model_views = joblib.load("view_count_predictor_model.pkl")

# Streamlit page setup
st.set_page_config(page_title='YouTube Video Stats Predictor', layout='centered')

st.title('YouTube Video Stats Predictor')
st.markdown(
    "Predict how many **views** and **likes** your video will get.\n\n"
    "Note: Predictions are **estimates**, not guarantees."
)

# Input URL
url = st.text_input("Enter a YouTube video URL:")

video_data = None
predicted_views = None
predicted_likes = None

if url:
    video_id = get_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
    else:
        video_data = fetch_video_data(video_id)
        if not video_data:
            st.error("❌ No data found for this video.")
        else:
            # Predictions
            predicted_views, predicted_likes = predict_views_and_likes(video_data)
            days_to_views, days_to_likes = estimate_days_to_reach(
                video_data, predicted_views, predicted_likes
            )

            # Layout: 3 Columns
            left, center, right = st.columns([1, 2, 1])

            # Left column: Current stats
            with left:
                st.subheader(" Current Stats")
                st.write(f"**Views:** {video_data['viewCount']}")
                st.write(f"**Likes:** {video_data['likeCount']}")
                st.write(f"**Published:** {video_data['publishedAt']}")
                st.write(f"**Days Since Published:** {video_data['days']}")

            # Center column: Thumbnail & Details
            with center:
                st.image(video_data["thumbnail"], caption=video_data["title"], use_container_width=True)
                st.markdown(f"**Title:** {video_data['title']}")
                st.markdown(f"**Description:** {video_data['description']}")
                st.markdown(f"[Watch Video](https://www.youtube.com/watch?v={video_id})")

            # Right column: Predicted stats
            with right:
                st.subheader(" Predicted Stats")
                st.write(f"**Views:** {predicted_views}")
                st.write(f"**Likes:** {predicted_likes}")
                st.write(f"**Days to Views Goal:** {days_to_views}")
                st.write(f"**Days to Likes Goal:** {days_to_likes}")

# Gemini AI advice generation with caching
@st.cache_data(show_spinner=False)
def generate_advice(prompt):
    response = model.generate_content(prompt)
    return response.text

st.markdown("### Tips to get the predicted views and likes")

if video_data and predicted_views is not None and predicted_likes is not None:
    if st.button("Advice me"):
        with st.spinner("Analyzing video and generating advice..."):
            prompt = f"""
You are an expert YouTube content strategist.

A YouTube video has the following details:
- Title: {video_data['title']}
- Description: {video_data['description']}
- Tags: {video_data.get('tags', [])}
- Duration in seconds: {convert_duration_to_seconds(video_data['duration'])}
- Category ID: {video_data['categoryId']}
- Current Views: {video_data['viewCount']}
- Current Likes: {video_data['likeCount']}
- Comments: {video_data.get('commentCount', 0)}
- Days Since Published: {video_data['days']}
- Predicted Views: {predicted_views}
- Predicted Likes: {predicted_likes}

Give me 5 practical and beginner-friendly tips that will help the creator reach the predicted views and likes faster. Focus on:
- Improving title, description, and tags
- Boosting engagement (likes, comments, watch time)
- Promoting or sharing the video
- Timing of posts
- Leveraging Shorts or other social media

Keep the advice simple, clear, and tailored to this video.
            """
            try:
                tips = generate_advice(prompt)
                st.success("Advice generated!")
                st.markdown(tips)
            except Exception as e:
                st.error(f"❌ Failed to generate advice: {str(e)}")
else:
    st.info("Enter a valid YouTube URL and get predictions first to receive advice.")

