import sys
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from reddit_data_collector import RedditDataCollector


def collect_comments(
    client_id, client_secret, user_agent,ms_file_path
):
    # Create an instance of RedditDataCollector
    collector = RedditDataCollector(client_id, client_secret, user_agent)

    # Collect comments from the r/microsoft subreddit
    ms_comments = collector.get_comments_from_submissions("microsoft", limit=100)
    #processed_ms_comments = preprocess_comments(ms_comments)

    # Collect comments from the r/amazon subreddit
    #am_comments = collector.get_comments_from_submissions("amazon", limit=10)
    #processed_am_comments = preprocess_comments(am_comments)

    # Saving the comments to a CSV file
    collector.save_comments_to_file(ms_comments, ms_file_path)
    #collector.save_comments_to_file(am_comments, am_file_path)


if __name__ == "__main__":
    collect_comments()
