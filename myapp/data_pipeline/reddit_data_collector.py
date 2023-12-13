import praw
import csv

class RedditDataCollector:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def get_comments_from_submissions(self, subreddit_name, limit=None):
        subreddit = self.reddit.subreddit(subreddit_name)
        comments = []
        
        # Replace `hot` with `new` or `top` if you want to get comments from those sections instead
        for submission in subreddit.hot(limit=limit):  # Here 'limit' controls the number of posts to fetch
            submission.comments.replace_more(limit=0)  # This line expands all comments
            for comment in submission.comments.list():
                comments.append(comment.body)
        
        return comments

    def save_comments_to_file(self, comments, file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_ALL)
            writer.writerow(['comment'])  # Header row
            for comment in comments:
                writer.writerow([comment])  # Write comment rows


  
