import pandas as pd
import numpy as np
from generate_wordclouds import get_global_north_countries, classify_country

def get_state(label):
    # Warm Tones (Impact/Suffering) -> State B (Impact/Disagreement/Negative)
    warm_labels = {
        'injured_or_dead_people', 
        'displaced_people_and_evacuations', 
        'not_humanitarian', 
        'infrastructure_and_utility_damage',
        'missing_or_found_people'
    }
    # Cool Tones (Response/Solidarity) -> State A (Action/Agreement/Positive)
    cool_labels = {
        'caution_and_advice', 
        'rescue_volunteering_or_donation_effort', 
        'sympathy_and_support'
    }
    # Neutral Tones -> State C (Info/Neutral)
    neutral_labels = {
        'other_relevant_information', 
        'requests_or_urgent_needs'
    }
    
    if label in warm_labels: return 'State B (Impact)'
    if label in cool_labels: return 'State A (Action)'
    return 'State C (Info)'

def main():
    print("Loading data...")
    df = pd.read_csv('crisisnlp_combined_dataset.csv')
    
    # Preprocessing
    print("Classifying regions...")
    global_north_list = get_global_north_countries()
    df['global_class'] = df['country'].apply(lambda x: classify_country(x, global_north_list))
    
    print("Assigning Behavioral States...")
    df['behavior_state'] = df['class_label'].apply(get_state)
    
    # 1. Antecedent: Economic Status (Region)
    print("\n--- Antecedent: Economic Status (Region) ---")
    region_stats = df.groupby('global_class')['behavior_state'].value_counts(normalize=True).unstack()
    print(region_stats * 100)
    
    # 2. Daily Transition Analysis
    # Ensure date handling
    # The dataset has 'disaster_start_date' and 'disaster_end_date' and 'tweet_time_utc'??
    # Let's check columns first, but assuming 'tweet_time_utc' exists from previous output.
    # If not, we might need to rely on 'dispute' or just ignore time if granular data is missing.
    # Checking previous output: "Columns: Index(['tweet_id'.... 'tweet_time_utc']" -> It exists!
    
    print("\n--- Temporal Analysis (Daily Probabilities) ---")
    # Clean date
    # Some dates might be diverse formats, try coerce
    df['date'] = pd.to_datetime(df['tweet_time_utc'], errors='coerce').dt.date
    
    # Filter out NaT
    time_df = df.dropna(subset=['date'])
    
    # Calculate daily probs per region
    # We want to see how State A (Action) probability changes over time for North vs South
    
    for region in ['Global North', 'Global South']:
        print(f"\nTime Analysis for {region}:")
        region_df = time_df[time_df['global_class'] == region]
        if region_df.empty:
            print("No data.")
            continue
            
        daily_counts = region_df.groupby(['date', 'behavior_state']).size().unstack(fill_value=0)
        # Normalize row-wise to get probabilities
        daily_probs = daily_counts.div(daily_counts.sum(axis=1), axis=0)
        
        # Print first few and summary
        print(daily_probs.head(10))
        print("...")
        print("Average Daily Probabilities:")
        print(daily_probs.mean())

if __name__ == "__main__":
    main()
