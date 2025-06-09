from predict import get_video_id,fetch_video_data,predict_views_and_likes,model_likes,model_views,show_comparison,estimate_days_to_reach



def main():
    url = input("Enter any Youtube url:")
    video_id = get_video_id(url)
    if not video_id:
        print("invalid url")
        return

    fetch_id = fetch_video_data(video_id)
    if not fetch_id:
        print("no data found")
        return
    predicted_views,predicted_likes = predict_views_and_likes(model_likes,model_views,fetch_id)
    days_to_view,days_to_like = estimate_days_to_reach(fetch_id , predicted_views,predicted_likes)
    show_comparison(days_to_view,days_to_like,predicted_likes,predicted_views,fetch_id)


if __name__ == "__main__":
    main()

    
