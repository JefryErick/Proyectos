import random
from datetime import datetime, timedelta

def generate_twitter_dataset():
    print("Generando dataset Twitter simulado...")
    num_users = 5000
    num_edges = 14000
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010, 3, 31)

    def random_date(start, end):
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

    with open("data/twitter-simulado.csv", "w") as f:
        f.write("user_id,follower_id,timestamp\n")
        for _ in range(num_edges):
            user_id = random.randint(1, num_users)
            follower_id = random.randint(1, num_users)
            while follower_id == user_id:
                follower_id = random.randint(1, num_users)
            timestamp = random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{user_id},{follower_id},{timestamp}\n")
    
    print("✅ Dataset Twitter generado: data/twitter-simulado.csv")

def generate_digg_dataset():
    print("Generando dataset Digg simulado...")
    num_users = 3000
    num_edges = 60000  # Digg tiene más interacciones
    start_date = datetime(2009, 6, 1)
    end_date = datetime(2009, 7, 31)

    def random_date(start, end):
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

    with open("data/digg-simulado.csv", "w") as f:
        f.write("user_id,friend_id,timestamp\n")  # Encabezado CSV
        for _ in range(num_edges):
            user_id = random.randint(1, num_users)
            friend_id = random.randint(1, num_users)
            while friend_id == user_id:
                friend_id = random.randint(1, num_users)
            timestamp = int(random_date(start_date, end_date).timestamp())
            f.write(f"{user_id},{friend_id},{timestamp}\n")  # Usar comas en lugar de tabs
    
    print("✅ Dataset Digg generado: data/digg-simulado.csv")

def generate_memetracker_dataset():
    print("Generando dataset Memetracker simulado...")
    num_phrases = 500
    num_sites = 200
    num_entries = 10000
    start_date = datetime(2008, 9, 1)
    end_date = datetime(2008, 12, 31)

    phrases = [f"phrase_{i}" for i in range(1, num_phrases+1)]
    sites = [f"site_{i}" for i in range(1, num_sites+1)]

    def random_date(start, end):
        return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

    with open("data/memetracker-simulado.csv", "w") as f:
        f.write("phrase,site,timestamp,frequency\n")  # Encabezado CSV
        for _ in range(num_entries):
            phrase = random.choice(phrases)
            site = random.choice(sites)
            timestamp = random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S")
            frequency = random.randint(1, 100)
            f.write(f"{phrase},{site},{timestamp},{frequency}\n")  # Usar comas en lugar de tabs
    
    print("✅ Dataset Memetracker generado: data/memetracker-simulado.csv")

if __name__ == "__main__":
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    
    generate_twitter_dataset()
    generate_digg_dataset()
    generate_memetracker_dataset()