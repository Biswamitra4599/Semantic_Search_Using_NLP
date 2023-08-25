import re

def tweet_cleaner(tt):
    clean_tweet=re.sub("@[A-Za-z0-9_]+","",tt)
    clean_tweet=re.sub("#[A-Za-z0-9_]+","",clean_tweet)
    clean_tweet=clean_tweet.lstrip(" ")
    return clean_tweet



tweet= "@nil @dar    Hi Buddy Great Day In Goa #Holiday #fun"
c_tweet=tweet_cleaner(tweet)
print(tweet)
print(c_tweet)
