import pandas as pd

def load_data():
    data = [
        {"id": 1, "text": "Stock markets rallied today as investors gained confidence."},
        {"id": 2, "text": "Scientists discovered a new exoplanet that may support life."},
        {"id": 3, "text": "The latest smartphone features AI-enhanced photography."},
        {"id": 4, "text": "Governments are working on climate change policies."},
        {"id": 5, "text": "Advancements in quantum computing are accelerating innovation."},
    ]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_data()
    print(df)
