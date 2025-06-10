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
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import asyncio

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load environment variables
load_dotenv()  

# Streamlit page setup
st.set_page_config(
    page_title='YouTube Video Stats Predictor', 
    layout='centered',
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.model_likes = None
    st.session_state.model_views = None

# Load models with error handling
@st.cache_resource
def load_ml_models():
    try:
        model_likes = joblib.load("like_count_predictor_model.pkl")
        model_views = joblib.load("view_count_predictor_model.pkl")
        return model_likes, model_views
    except Exception as e:
        st.error(f"❌ Error loading ML models: {str(e)}")
        st.info("Please ensure your model files are compatible with the current scikit-learn version.")
        return None, None

# Initialize Gemini AI with error handling
def initialize_gemini():
    try:
        gemini_api_key = os.getenv("GENAI_API_KEY")
        if not gemini_api_key:
            st.error("❌ GENAI_API_KEY not found in environment variables")
            return None
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('models/gemini-2.0-flash-lite-preview')
        return model
    except Exception as e:
        st.error(f"❌ Error initializing Gemini AI: {str(e)}")
        return None

# Load models and initialize Gemini
if not st.session_state.models_loaded:
    with st.spinner("Loading ML models..."):
        model_likes, model_views = load_ml_models()
        if model_likes and model_views:
            st.session_state.model_likes = model_likes
            st.session_state.model_views = model_views
            st.session_state.models_loaded = True

gemini_model = initialize_gemini()

st.title('YouTube Video Stats Predictor')
st.markdown(
    "Predict how many **views** and **likes** your video will get.\n\n"
    "Note: Predictions are **estimates**, not guarantees."
)

# Check if models are loaded
if not st.session_state.models_loaded:
    st.warning("⚠️ ML models could not be loaded. Please check your model files.")
    st.stop()

# Input URL
url = st.text_input("Enter a YouTube video URL:")

video_data = None
predicted_views = None
predicted_likes = None

if url:
    try:
        video_id = get_video_id(url)
        if not video_id:
            st.error("❌ Invalid YouTube URL.")
        else:
            with st.spinner("Fetching video data..."):
                video_data = fetch_video_data(video_id)
                
            if not video_data:
                st.error("❌ No data found for this video.")
            else:
                # Predictions with error handling
                try:
                    with st.spinner("Making predictions..."):
                        predicted_views, predicted_likes = predict_views_and_likes(video_data)
                        days_to_views, days_to_likes = estimate_days_to_reach(
                            video_data, predicted_views, predicted_likes
                        )
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.stop()

                # Layout: 3 Columns
                left, center, right = st.columns([1, 2, 1])

                # Left column: Current stats
                with left:
                    st.subheader("📊 Current Stats")
                    try:
                        st.write(f"**Views:** {video_data.get('viewCount', 'N/A')}")
                        st.write(f"**Likes:** {video_data.get('likeCount', 'N/A')}")
                        st.write(f"**Published:** {video_data.get('publishedAt', 'N/A')}")
                        st.write(f"**Days Since Published:** {video_data.get('days', 'N/A')}")
                    except Exception as e:
                        st.error(f"Error displaying current stats: {str(e)}")

                # Center column: Thumbnail & Details
                with center:
                    try:
                        if video_data.get("thumbnail"):
                            st.image(
                                video_data["thumbnail"], 
                                caption=video_data.get("title", "No title"), 
                                use_container_width=True
                            )
                        st.markdown(f"**Title:** {video_data.get('title', 'No title')}")
                        
                        # Truncate description if too long
                        description = video_data.get('description', 'No description')
                        if len(description) > 200:
                            description = description[:200] + "..."
                        st.markdown(f"**Description:** {description}")
                        
                        st.markdown(f"[Watch Video](https://www.youtube.com/watch?v={video_id})")
                    except Exception as e:
                        st.error(f"Error displaying video details: {str(e)}")

                # Right column: Predicted stats
                with right:
                    st.subheader("🔮 Predicted Stats")
                    try:
                        st.write(f"**Views:** {predicted_views if predicted_views is not None else 'N/A'}")
                        st.write(f"**Likes:** {predicted_likes if predicted_likes is not None else 'N/A'}")
                        st.write(f"**Days to Views Goal:** {days_to_views if 'days_to_views' in locals() else 'N/A'}")
                        st.write(f"**Days to Likes Goal:** {days_to_likes if 'days_to_likes' in locals() else 'N/A'}")
                    except Exception as e:
                        st.error(f"Error displaying predictions: {str(e)}")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")

# Gemini AI advice generation with caching and error handling
@st.cache_data(show_spinner=False)
def generate_advice(prompt):
    try:
        if not gemini_model:
            raise Exception("Gemini AI model not initialized")
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Failed to generate advice: {str(e)}")

st.markdown("### 💡 Tips to get the predicted views and likes")

if video_data and predicted_views is not None and predicted_likes is not None:
    if st.button("🤖 Get AI Advice", type="primary"):
        if not gemini_model:
            st.error("❌ Gemini AI is not available. Please check your API key.")
        else:
            with st.spinner("Analyzing video and generating advice..."):
                try:
                    # Safely get video data with defaults
                    title = video_data.get('title', 'No title')
                    description = video_data.get('description', 'No description')
                    tags = video_data.get('tags', [])
                    duration = video_data.get('duration', 'PT0S')
                    category_id = video_data.get('categoryId', 'Unknown')
                    view_count = video_data.get('viewCount', 0)
                    like_count = video_data.get('likeCount', 0)
                    comment_count = video_data.get('commentCount', 0)
                    days = video_data.get('days', 0)
                    
                    # Convert duration safely
                    try:
                        duration_seconds = convert_duration_to_seconds(duration)
                    except:
                        duration_seconds = 0
                    
                    prompt = f"""
You are an expert YouTube content strategist.

A YouTube video has the following details:
- Title: {title}
- Description: {description[:500]}...  
- Tags: {tags[:10]}  
- Duration in seconds: {duration_seconds}
- Category ID: {category_id}
- Current Views: {view_count}
- Current Likes: {like_count}
- Comments: {comment_count}
- Days Since Published: {days}
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
                    
                    tips = generate_advice(prompt)
                    st.success("✅ Advice generated!")
                    st.markdown(tips)
                    
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                    st.info("Please try again or check your internet connection.")
else:
    st.info("📝 Enter a valid YouTube URL and get predictions first to receive advice.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Gemini AI")