import os
from steam_web_api import Steam

KEY = "91342BE6AC79D6639C5E6D3B5AEF8617"
steam = Steam(KEY)
stewie = steam.users.search_user("FishBone000")
print(stewie)
print(steam.users.get_user_recently_played_games(stewie['player']['steamid']))