import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

def load_data(filepath):
    """Loads the dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    """
    Cleans tweet text by removing URLs, handles, and special characters.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\w+', ' ', text)
    # Split camelCase words (e.g. HurricaneHarvey -> Hurricane Harvey)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Remove hashtags (keeping the word)
    text = re.sub(r'#', '', text)
    # Remove special characters and numbers (replace with space to avoid concatenation)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove short words (length < 2)
    text = ' '.join([w for w in text.split() if len(w) >= 2])
    return text

def get_global_north_countries():
    """
    Returns a list of countries considered 'Global North' (Rich).
    This is a proxy for GDP-based classification.
    """
    return [
        'United States', 'USA', 'Canada', 
        'United Kingdom', 'UK', 'Great Britain', 'England',
        'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium',
        'Sweden', 'Norway', 'Denmark', 'Finland', 'Switzerland', 'Austria',
        'Ireland', 'Luxembourg',
        'Australia', 'New Zealand',
        'Japan', 'South Korea', 'Singapore', 'Israel'
    ]

def classify_country(country_name, global_north_list):
    """
    Classifies a country as 'Global North' or 'Global South'.
    """
    if not isinstance(country_name, str):
        return 'Unknown'
    
    # Simple check if the country name (or part of it) is in the list
    # Normalize to title case for comparison
    country_norm = country_name.strip()
    
    if country_norm in global_north_list:
        return 'Global North'
    
    # Fallback checks (e.g. if 'USA' is 'United States of America')
    for gn_country in global_north_list:
        if gn_country.lower() in country_norm.lower():
            return 'Global North'
            
    return 'Global South'

class WeightedColorFunc(object):
    """
    Assigns colors to words based on the weighted average of their occurrence in different class labels.
    Warm Tones: injured_or_dead, displaced, infrastructure, etc.
    Neutral Tones: other, requests.
    Cool Tones: caution, rescue, sympathy.
    """
    def __init__(self, data_df):
        self.word_counts = {} # word -> {category: count}
        
        # Define categories and their representative RGB colors
        self.WARM_COLOR = (255, 140, 0)   # Dark Orange
        self.NEUTRAL_COLOR = (34, 139, 34) # Forest Green
        self.COOL_COLOR = (65, 105, 225)  # Royal Blue
        
        self.warm_labels = {
            'injured_or_dead_people', 
            'displaced_people_and_evacuations', 
            'not_humanitarian', 
            'infrastructure_and_utility_damage',
            'missing_or_found_people'
        }
        self.neutral_labels = {
            'other_relevant_information', 
            'requests_or_urgent_needs'
        }
        self.cool_labels = {
            'caution_and_advice', 
            'rescue_volunteering_or_donation_effort', 
            'sympathy_and_support'
        }

        # Build frequency map
        print("Building word frequency map for coloring...")
        for _, row in data_df.iterrows():
            label = row['class_label']
            text = str(row['cleaned_text'])
            if not text: continue
            
            words = text.split()
            for word in words:
                if word not in self.word_counts:
                    self.word_counts[word] = {'warm': 0, 'neutral': 0, 'cool': 0}
                
                if label in self.warm_labels:
                    self.word_counts[word]['warm'] += 1
                elif label in self.neutral_labels:
                    self.word_counts[word]['neutral'] += 1
                elif label in self.cool_labels:
                    self.word_counts[word]['cool'] += 1
                    
    def __call__(self, word, font_size, position, orientation, random_state=None, **kwargs):
        word_lower = word.lower()
        if word_lower not in self.word_counts:
            return "rgb(128, 128, 128)" # Grey default
            
        counts = self.word_counts[word_lower]
        total = counts['warm'] + counts['neutral'] + counts['cool']
        
        if total == 0:
            return "rgb(128, 128, 128)"
            
        # Weighted average
        r = (counts['warm'] * self.WARM_COLOR[0] + 
             counts['neutral'] * self.NEUTRAL_COLOR[0] + 
             counts['cool'] * self.COOL_COLOR[0]) / total
             
        g = (counts['warm'] * self.WARM_COLOR[1] + 
             counts['neutral'] * self.NEUTRAL_COLOR[1] + 
             counts['cool'] * self.COOL_COLOR[1]) / total
             
        b = (counts['warm'] * self.WARM_COLOR[2] + 
             counts['neutral'] * self.NEUTRAL_COLOR[2] + 
             counts['cool'] * self.COOL_COLOR[2]) / total
             
        return f"rgb({int(r)}, {int(g)}, {int(b)})"


def generate_word_cloud(text_data, title, output_filename, color_func_data=None):
    """Generates and saves a word cloud image."""
    if not text_data.strip():
        print(f"No text data found for {title}. Skipping word cloud generation.")
        return

    stopwords = set(STOPWORDS)
    # Add custom stopwords if necessary
    stopwords.update(["rt", "via", "amp"])

    # Initialize color function if data is provided
    color_func = None
    if color_func_data is not None:
        color_func = WeightedColorFunc(color_func_data)

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=stopwords,
        min_font_size=10,
        color_func=color_func
    ).generate(text_data)

    plt.figure(figsize=(10, 5), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear") # Added bilinear for smoother look
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title)
    
    plt.savefig(output_filename)
    print(f"Saved word cloud to {output_filename}")
    plt.close()

def main():
    dataset_path = 'crisisnlp_combined_dataset.csv'
    df = load_data(dataset_path)
    
    if df is None:
        return

    # Check available columns
    print("Columns:", df.columns)
    
    # 1. Clean Text
    print("Cleaning text...")
    df['cleaned_text'] = df['tweet_text'].apply(clean_text)
    print("Sample cleaned text:")
    print(df['cleaned_text'].head())
    
    # Verify camelCase splitting
    test_str = "#HurricaneHarvey relief"
    print(f"\nTest camelCase cleaning: '{test_str}' -> '{clean_text(test_str)}'")
    
    # 2. Classify Countries
    print("Classifying countries...")
    global_north_list = get_global_north_countries()
    df['global_class'] = df['country'].apply(lambda x: classify_country(x, global_north_list))
    
    print("Country Classification Summary:")
    print(df['global_class'].value_counts())
    
    # Debug: Print some mappings
    print("\nSample Country Mappings:")
    print(df[['country', 'global_class']].drop_duplicates().head(20))

    # 3. Separate Data
    north_df = df[df['global_class'] == 'Global North']
    south_df = df[df['global_class'] == 'Global South']
    
    north_text = " ".join(north_df['cleaned_text'])
    south_text = " ".join(south_df['cleaned_text'])
    
    # 4. Generate Word Clouds
    print("\nGenerating Global North Word Cloud...")
    generate_word_cloud(north_text, "Global North (Rich) Sentiment", "wordcloud_global_north.png", color_func_data=north_df)
    
    print("\nGenerating Global South Word Cloud...")
    generate_word_cloud(south_text, "Global South (Poor) Sentiment", "wordcloud_global_south.png", color_func_data=south_df)

    print("\nDone.")

if __name__ == "__main__":
    main()
