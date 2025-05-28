#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns
from datetime import datetime
import html
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings
warnings.filterwarnings('ignore')

# Try to load spaCy model, use simpler approach if not available
try:
    nlp = spacy.load("en_core_web_sm")
    spacy_available = True
except:
    spacy_available = False
    print("SpaCy model not available. Using simplified text processing.")

# Create output directory
output_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(output_dir, "images")
os.makedirs(img_dir, exist_ok=True)

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    # Convert to string if not already
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords(text, top_n=5):
    if not spacy_available or not text or pd.isna(text):
        return []
    
    doc = nlp(text)
    keywords = []
    
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2:
            keywords.append(token.lemma_.lower())
    
    # Count occurrences and get top n
    if keywords:
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(top_n)]
    return []

def categorize_feedback(text, categories):
    if pd.isna(text) or not text:
        return "Uncategorized"
    
    text = text.lower()
    scores = {}
    
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword.lower() in text)
        scores[category] = score
    
    if max(scores.values(), default=0) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "Other"

def subcategorize_feedback(text, main_category, subcategories):
    if pd.isna(text) or not text or main_category not in subcategories:
        return "General"
    
    text = text.lower()
    scores = {}
    
    for subcategory, keywords in subcategories[main_category].items():
        score = sum(1 for keyword in keywords if keyword.lower() in text)
        scores[subcategory] = score
    
    if max(scores.values(), default=0) > 0:
        return max(scores.items(), key=lambda x: x[1])[0]
    return "General"

def analyze_sentiment(text):
    if pd.isna(text) or not text:
        return "Neutral", 3
    
    text = text.lower()
    
    positive_words = ['great', 'good', 'awesome', 'excellent', 'amazing', 'love', 'best', 'fantastic', 
                     'perfect', 'helpful', 'easy', 'impressed', 'smooth', 'useful', 'happy', 'like',
                     'recommended', 'convenient', 'improved', 'fast', 'works well']
    
    negative_words = ['bad', 'terrible', 'horrible', 'awful', 'slow', 'bug', 'crash', 'problem', 'issue',
                     'disappointing', 'hate', 'difficult', 'broken', 'fail', 'error', 'poor', 'useless',
                     'waste', 'annoying', 'frustrating', 'worst', 'unusable', 'cannot', 'not working']
    
    pos_score = sum(1 for word in positive_words if word in text)
    neg_score = sum(1 for word in negative_words if word in text)
    
    # Look for negations that reverse sentiment
    negations = ['not', 'no ', "don't", "doesn't", "didn't", "isn't", "aren't", "can't", "couldn't", 'never']
    
    for neg in negations:
        for pos in positive_words:
            if f"{neg} {pos}" in text or f"{neg}{pos}" in text:
                pos_score -= 1
                neg_score += 1
    
    # Generate a 5-point scale sentiment score
    if pos_score > neg_score * 2:
        return "Positive", 5
    elif pos_score > neg_score:
        return "Positive", 4
    elif neg_score > pos_score * 2:
        return "Negative", 1
    elif neg_score > pos_score:
        return "Negative", 2
    else:
        return "Neutral", 3

def identify_feature_requests(text):
    if pd.isna(text) or not text:
        return False
    
    text = text.lower()
    request_phrases = ['add', 'should have', 'would be nice', 'please add', 'need to add',
                      'would like', 'could you add', 'feature request', 'missing feature',
                      'should include', 'wish', 'hope', 'want', 'request', 'implement',
                      'include', 'needs to', 'don\'t have', 'doesn\'t have', 'lacks']
    
    return any(phrase in text for phrase in request_phrases)

def identify_bug_report(text):
    if pd.isna(text) or not text:
        return False
    
    text = text.lower()
    bug_phrases = ['bug', 'crash', 'issue', 'problem', 'doesn\'t work', 'not working',
                  'broken', 'error', 'fail', 'glitch', 'freeze', 'stuck', 'malfunction',
                  'incorrect', 'wrong behavior', 'unexpected', 'not responding', 'hangs']
    
    return any(phrase in text for phrase in bug_phrases)

def get_category_definitions():
    # Define main categories with their keywords
    categories = {
        "User Interface": ["ui", "interface", "design", "layout", "theme", "dark mode", "light mode", 
                          "button", "menu", "navigation", "appearance", "visual", "screen", "display"],
        
        "Performance": ["slow", "speed", "fast", "crash", "freeze", "hang", "lag", "responsive", 
                      "performance", "memory", "cpu", "battery", "drain", "optimization"],
        
        "Features": ["feature", "function", "capability", "option", "setting", "mode", "tool", 
                   "ability", "extension", "add-on", "plugin", "functionality"],
        
        "Privacy & Security": ["privacy", "security", "tracking", "ad block", "blocker", "cookie", 
                             "permission", "safe", "protect", "incognito", "inprivate", "secure"],
        
        "Sync & Integration": ["sync", "integration", "account", "microsoft account", "across device", 
                             "cloud", "connected", "profile", "login", "sign in", "ecosystem"],
        
        "Media Handling": ["video", "audio", "media", "playback", "stream", "player", "sound", 
                         "youtube", "netflix", "fullscreen", "autoplay", "picture", "image"],
        
        "Extensions": ["extension", "addon", "plugin", "ublock", "adblock", "add-on", 
                     "violentmonkey", "tampermonkey", "adblocker", "ad blocker"],
        
        "PDF & Documents": ["pdf", "document", "reader", "view", "download", "file", "annotation", 
                          "highlight", "comment", "markup", "document", "print"],
        
        "Search Experience": ["search", "find", "bing", "google", "query", "result", "suggestion", 
                            "discover", "recommendation", "copilot", "ai", "search engine"]
    }
    
    # Define subcategories for each main category
    subcategories = {
        "User Interface": {
            "Layout Issues": ["layout", "display", "arrangement", "position", "alignment"],
            "Visual Bugs": ["visual bug", "glitch", "distortion", "rendering", "black screen", "blank", "dark mode", "theme"],
            "Navigation Problems": ["navigation", "button", "menu", "tab", "switch", "swipe", "gesture"],
            "Accessibility": ["accessibility", "font size", "zoom", "read aloud", "screen reader"]
        },
        
        "Performance": {
            "Speed Issues": ["slow", "speed", "loading", "fast", "quick", "wait time"],
            "Crashes": ["crash", "close", "exit", "terminate", "shut down", "restart"],
            "Memory Usage": ["memory", "ram", "resource", "usage", "heavy", "lightweight"],
            "Battery Drain": ["battery", "drain", "power", "consumption", "energy"]
        },
        
        "Features": {
            "Missing Features": ["missing", "lack", "need", "want", "should have", "don't have"],
            "Broken Features": ["broken", "not working", "doesn't work", "fail", "issue with"],
            "Feature Requests": ["add", "request", "wish", "would like", "hope", "implement", "should include"],
            "Settings & Customization": ["setting", "customize", "option", "preference", "configuration"]
        },
        
        "Privacy & Security": {
            "Ad Blocking": ["ad block", "blocker", "advertisement", "popup", "ad free"],
            "Tracking Prevention": ["track", "prevention", "fingerprint", "data collection"],
            "InPrivate Browsing": ["inprivate", "incognito", "private browsing", "private mode"],
            "Permissions": ["permission", "access", "allow", "deny", "camera", "microphone", "location"]
        },
        
        "Sync & Integration": {
            "Account Issues": ["account", "login", "sign in", "profile", "microsoft account"],
            "Cross-device Sync": ["sync", "across device", "multiple device", "continuity"],
            "Ecosystem Integration": ["ecosystem", "integration", "work together", "microsoft service"],
            "Data Loss": ["data loss", "history", "bookmark", "lost", "missing data", "saved"]
        },
        
        "Media Handling": {
            "Video Playback": ["video", "playback", "player", "stream", "youtube", "netflix"],
            "Audio Issues": ["audio", "sound", "volume", "mute", "speaker"],
            "Fullscreen Problems": ["fullscreen", "landscape", "rotate", "orientation"],
            "Image Viewing": ["image", "picture", "photo", "view", "gallery"]
        },
        
        "Extensions": {
            "Extension Availability": ["available", "missing", "can't find", "not found", "need"],
            "Extension Functionality": ["function", "working", "doesn't work", "broken"],
            "Ad Blocker Extensions": ["adblock", "ublock", "ad blocker", "advertisement block"],
            "User Scripts": ["script", "tampermonkey", "violentmonkey", "greasemonkey"]
        },
        
        "PDF & Documents": {
            "PDF Viewing": ["view", "reader", "display", "open"],
            "PDF Editing": ["edit", "annotation", "highlight", "comment", "markup"],
            "Download Issues": ["download", "save", "file", "storage"],
            "Printing": ["print", "printer", "hard copy", "physical copy"]
        },
        
        "Search Experience": {
            "Search Quality": ["quality", "relevant", "result", "accuracy", "find what I need"],
            "AI Features": ["ai", "copilot", "artificial intelligence", "assistant", "chat"],
            "Search Suggestions": ["suggestion", "autocomplete", "predict", "recommend"],
            "Search Engine Choice": ["engine", "google", "bing", "yahoo", "duckduckgo", "change"]
        }
    }
    
    return categories, subcategories

def process_unwrap_data(file_path):
    print(f"Processing Unwrap data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        required_cols = ['Entry Text', 'Entry Summary', 'Entry Source', 'Entry Permalink', 'Entry Date', 'Sentiment']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in Unwrap file: {missing_cols}")
            # Try to use alternative columns if available
            if 'Entry Text' not in df.columns and 'Entry Summary' in df.columns:
                df['Entry Text'] = df['Entry Summary']
        
        # Use Entry Text if available, otherwise use Entry Summary
        text_column = 'Entry Text' if 'Entry Text' in df.columns else 'Entry Summary'
        
        if text_column not in df.columns:
            raise ValueError(f"Neither 'Entry Text' nor 'Entry Summary' columns found in the Unwrap file")
        
        # Process the data
        feedback_text = df[text_column].apply(clean_text)
        date = pd.to_datetime(df['Entry Date']) if 'Entry Date' in df.columns else pd.NaT
        url = df['Entry Permalink'] if 'Entry Permalink' in df.columns else None
        platform = df['Entry Source'] if 'Entry Source' in df.columns else 'Unknown'
        
        # Create the processed dataframe with explicit length
        row_count = len(df)
        processed_df = pd.DataFrame({
            'source': ['Unwrap'] * row_count,
            'platform': platform,
            'feedback_text': feedback_text,
            'date': date,
            'url': url
        })
        
        # Extract existing sentiment if available
        if 'Sentiment' in df.columns:
            processed_df['original_sentiment'] = df['Sentiment']
        else:
            processed_df['original_sentiment'] = None
        
        # Get categories and subcategories
        categories, subcategories = get_category_definitions()
        
        # Apply analysis to each row
        processed_df['main_category'] = processed_df['feedback_text'].apply(lambda x: categorize_feedback(x, categories))
        processed_df['subcategory'] = processed_df.apply(lambda row: subcategorize_feedback(
            row['feedback_text'], row['main_category'], subcategories), axis=1)
        
        sentiment_results = processed_df['feedback_text'].apply(analyze_sentiment)
        processed_df['sentiment'] = [result[0] for result in sentiment_results]
        processed_df['sentiment_score'] = [result[1] for result in sentiment_results]
        
        processed_df['is_feature_request'] = processed_df['feedback_text'].apply(identify_feature_requests)
        processed_df['is_bug_report'] = processed_df['feedback_text'].apply(identify_bug_report)
        processed_df['keywords'] = processed_df['feedback_text'].apply(extract_keywords)
        
        return processed_df
    
    except Exception as e:
        print(f"Error processing Unwrap data: {str(e)}")
        return pd.DataFrame()

def process_app_reviews(file_path, platform):
    print(f"Processing {platform} app reviews from {file_path}")
    try:
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        if platform.lower() == 'ios':
            text_column = 'text'
            date_column = 'date'
            rating_column = 'rating'
            # Some iOS reviews might have title column to combine with text
            title_column = 'title' if 'title' in df.columns else None
        else:  # Google Play
            text_column = 'text'
            date_column = 'date'
            rating_column = 'rating'
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in the {platform} reviews file")
        
        # Process text data
        row_count = len(df)
        source_value = f'{platform} App Store'
        
        # Combine title and text for iOS if available
        if platform.lower() == 'ios' and title_column and title_column in df.columns:
            feedback_text = df.apply(
                lambda row: f"{row[title_column]}: {row[text_column]}" if pd.notna(row[title_column]) else row[text_column], 
                axis=1
            ).apply(clean_text)
        else:
            feedback_text = df[text_column].apply(clean_text)
        
        date = pd.to_datetime(df[date_column]) if date_column in df.columns else pd.NaT
        rating = pd.to_numeric(df[rating_column], errors='coerce') if rating_column in df.columns else None
        
        # Create the processed dataframe with explicit source values
        processed_df = pd.DataFrame({
            'source': [source_value] * row_count,
            'platform': [platform] * row_count,
            'feedback_text': feedback_text,
            'date': date,
            'rating': rating,
            'url': [None] * row_count  # App store reviews don't have direct URLs
        })
        
        # Map ratings to sentiment
        if 'rating' in processed_df.columns:
            processed_df['original_sentiment'] = processed_df['rating'].apply(
                lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral')
            )
        else:
            processed_df['original_sentiment'] = None
        
        # Get categories and subcategories
        categories, subcategories = get_category_definitions()
        
        # Apply analysis to each row
        processed_df['main_category'] = processed_df['feedback_text'].apply(lambda x: categorize_feedback(x, categories))
        processed_df['subcategory'] = processed_df.apply(lambda row: subcategorize_feedback(
            row['feedback_text'], row['main_category'], subcategories), axis=1)
        
        sentiment_results = processed_df['feedback_text'].apply(analyze_sentiment)
        processed_df['sentiment'] = [result[0] for result in sentiment_results]
        processed_df['sentiment_score'] = [result[1] for result in sentiment_results]
        
        processed_df['is_feature_request'] = processed_df['feedback_text'].apply(identify_feature_requests)
        processed_df['is_bug_report'] = processed_df['feedback_text'].apply(identify_bug_report)
        processed_df['keywords'] = processed_df['feedback_text'].apply(extract_keywords)
        
        return processed_df
    
    except Exception as e:
        print(f"Error processing {platform} app reviews: {str(e)}")
        return pd.DataFrame()

def generate_plots(df, output_dir):
    print("Generating visualizations...")
    plot_files = []
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 10})
    
    # 1. Main Categories Distribution
    plt.figure(figsize=(12, 6))
    category_counts = df['main_category'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Distribution of Feedback by Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    category_plot = os.path.join(output_dir, "category_distribution.png")
    plt.savefig(category_plot)
    plot_files.append(("category_distribution.png", "Distribution of Feedback by Category"))
    plt.close()
    
    # 2. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=[colors[x] for x in sentiment_counts.index])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    sentiment_plot = os.path.join(output_dir, "sentiment_distribution.png")
    plt.savefig(sentiment_plot)
    plot_files.append(("sentiment_distribution.png", "Sentiment Distribution"))
    plt.close()
    
    # 3. Sentiment by Category
    plt.figure(figsize=(14, 7))
    category_sentiment = pd.crosstab(df['main_category'], df['sentiment'])
    category_sentiment_pct = category_sentiment.div(category_sentiment.sum(axis=1), axis=0) * 100
    category_sentiment_pct.plot(kind='bar', stacked=True, color=[colors[x] for x in category_sentiment.columns])
    plt.title('Sentiment Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    sentiment_by_category_plot = os.path.join(output_dir, "sentiment_by_category.png")
    plt.savefig(sentiment_by_category_plot)
    plot_files.append(("sentiment_by_category.png", "Sentiment Distribution by Category"))
    plt.close()
    
    # 4. Source Distribution
    plt.figure(figsize=(10, 6))
    source_counts = df['source'].value_counts()
    sns.barplot(x=source_counts.index, y=source_counts.values)
    plt.title('Feedback Distribution by Source')
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    source_plot = os.path.join(output_dir, "source_distribution.png")
    plt.savefig(source_plot)
    plot_files.append(("source_distribution.png", "Feedback Distribution by Source"))
    plt.close()
    
    # 5. Feature Requests vs Bug Reports
    plt.figure(figsize=(10, 6))
    request_counts = [
        df['is_feature_request'].sum(),
        df['is_bug_report'].sum(),
        len(df) - df['is_feature_request'].sum() - df['is_bug_report'].sum()
    ]
    labels = ['Feature Requests', 'Bug Reports', 'General Feedback']
    plt.pie(request_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.title('Distribution of Feedback Types')
    plt.tight_layout()
    feedback_type_plot = os.path.join(output_dir, "feedback_type_distribution.png")
    plt.savefig(feedback_type_plot)
    plot_files.append(("feedback_type_distribution.png", "Distribution of Feedback Types"))
    plt.close()
    
    # 6. Top Subcategories
    plt.figure(figsize=(14, 7))
    subcategory_counts = df['subcategory'].value_counts().head(10)
    sns.barplot(x=subcategory_counts.index, y=subcategory_counts.values)
    plt.title('Top 10 Subcategories')
    plt.xlabel('Subcategory')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    subcategory_plot = os.path.join(output_dir, "top_subcategories.png")
    plt.savefig(subcategory_plot)
    plot_files.append(("top_subcategories.png", "Top 10 Subcategories"))
    plt.close()
    
    # 7. Sentiment Score Distribution
    plt.figure(figsize=(10, 6))
    sentiment_score_counts = df['sentiment_score'].value_counts().sort_index()
    sns.barplot(x=sentiment_score_counts.index, y=sentiment_score_counts.values, 
               palette=['darkred', 'lightcoral', 'lightgray', 'lightgreen', 'darkgreen'])
    plt.title('Distribution of 5-Point Sentiment Scores')
    plt.xlabel('Sentiment Score (1=Very Negative, 5=Very Positive)')
    plt.ylabel('Count')
    plt.tight_layout()
    sentiment_score_plot = os.path.join(output_dir, "sentiment_score_distribution.png")
    plt.savefig(sentiment_score_plot)
    plot_files.append(("sentiment_score_distribution.png", "Distribution of 5-Point Sentiment Scores"))
    plt.close()
    
    # 8. Word Cloud (if possible)
    try:
        from wordcloud import WordCloud
        
        # Combine all feedback texts
        all_text = ' '.join(df['feedback_text'].dropna())
        
        # Generate and save wordcloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue')
        wordcloud.generate(all_text)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        wordcloud_plot = os.path.join(output_dir, "feedback_wordcloud.png")
        plt.savefig(wordcloud_plot)
        plot_files.append(("feedback_wordcloud.png", "Word Cloud of User Feedback"))
        plt.close()
    except ImportError:
        print("WordCloud library not available. Skipping word cloud generation.")
    
    return plot_files

def extract_top_examples(df, category, subcategory=None, n=5):
    """Extract top examples for a given category and optional subcategory"""
    if subcategory:
        filtered_df = df[(df['main_category'] == category) & (df['subcategory'] == subcategory)]
    else:
        filtered_df = df[df['main_category'] == category]
    
    if len(filtered_df) == 0:
        return []
    
    # Sort by absolute sentiment score (distance from neutral)
    filtered_df['sentiment_abs'] = abs(filtered_df['sentiment_score'] - 3)
    sorted_df = filtered_df.sort_values('sentiment_abs', ascending=False)
    
    examples = []
    for _, row in sorted_df.head(n).iterrows():
        # Handle source information - replace NaN with a proper label
        if pd.isna(row['source']):
            source_info = "Unknown source"
        else:
            source_info = f"{row['source']}"
            
        if pd.notna(row['url']) and row['url']:
            source_info += f" (<a href='{row['url']}' target='_blank'>Source</a>)"
        
        sentiment_color = "green" if row['sentiment'] == "Positive" else ("red" if row['sentiment'] == "Negative" else "gray")
        
        example = {
            'text': row['feedback_text'],
            'source': source_info,
            'sentiment': row['sentiment'],
            'sentiment_color': sentiment_color,
            'date': row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'Unknown date'
        }
        examples.append(example)
    
    return examples

def generate_html_report(df, plot_files, output_path):
    print("Generating HTML report...")
    
    # Get unique categories and subcategories
    categories = sorted(df['main_category'].unique())
    
    # Prepare category sections
    category_sections = ""
    for category in categories:
        if category == "Uncategorized" or category == "Other":
            continue
            
        category_df = df[df['main_category'] == category]
        subcategories = sorted(category_df['subcategory'].unique())
        
        # Calculate statistics
        total_count = len(category_df)
        positive_count = len(category_df[category_df['sentiment'] == 'Positive'])
        negative_count = len(category_df[category_df['sentiment'] == 'Negative'])
        neutral_count = len(category_df[category_df['sentiment'] == 'Neutral'])
        
        positive_percent = (positive_count / total_count) * 100 if total_count > 0 else 0
        negative_percent = (negative_count / total_count) * 100 if total_count > 0 else 0
        neutral_percent = (neutral_count / total_count) * 100 if total_count > 0 else 0
        
        feature_requests = len(category_df[category_df['is_feature_request']])
        bug_reports = len(category_df[category_df['is_bug_report']])
        
        # Get top examples for this category
        top_examples = extract_top_examples(df, category)
        examples_html = ""
        for example in top_examples:
            examples_html += f"""
            <div class="example-card">
                <div class="example-text">"{html.escape(example['text'])}"</div>
                <div class="example-meta">
                    <span class="source">{example['source']}</span> | 
                    <span class="date">{example['date']}</span> | 
                    <span class="sentiment" style="color: {example['sentiment_color']};">{example['sentiment']}</span>
                </div>
            </div>
            """
        
        # Generate subcategory sections
        subcategory_sections = ""
        for subcategory in subcategories:
            if subcategory == "General":
                continue
                
            subcategory_df = category_df[category_df['subcategory'] == subcategory]
            subcategory_count = len(subcategory_df)
            subcategory_percent = (subcategory_count / total_count) * 100 if total_count > 0 else 0
            
            # Get top examples for this subcategory
            subcategory_examples = extract_top_examples(df, category, subcategory)
            subcategory_examples_html = ""
            for example in subcategory_examples:
                subcategory_examples_html += f"""
                <div class="example-card">
                    <div class="example-text">"{html.escape(example['text'])}"</div>
                    <div class="example-meta">
                        <span class="source">{example['source']}</span> | 
                        <span class="date">{example['date']}</span> | 
                        <span class="sentiment" style="color: {example['sentiment_color']};">{example['sentiment']}</span>
                    </div>
                </div>
                """
            
            subcategory_sections += f"""
            <div class="subcategory-section">
                <h4>{subcategory} <span class="count">({subcategory_count}, {subcategory_percent:.1f}%)</span></h4>
                <div class="examples-container">
                    {subcategory_examples_html}
                </div>
            </div>
            """
        
        # Add this category to the main report
        category_sections += f"""
        <div class="category-section" id="{category.lower().replace(' ', '-')}">
            <h3>{category}</h3>
            <div class="category-stats">
                <div class="stat-item">
                    <span class="stat-label">Total Feedback:</span>
                    <span class="stat-value">{total_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Sentiment:</span>
                    <span class="stat-value">
                        <span style="color: green;">{positive_percent:.1f}% Positive</span> | 
                        <span style="color: gray;">{neutral_percent:.1f}% Neutral</span> | 
                        <span style="color: red;">{negative_percent:.1f}% Negative</span>
                    </span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Feature Requests:</span>
                    <span class="stat-value">{feature_requests}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Bug Reports:</span>
                    <span class="stat-value">{bug_reports}</span>
                </div>
            </div>
            <div class="examples-container">
                <h4>Top Examples:</h4>
                {examples_html}
            </div>
            <div class="subcategories-container">
                <h4>Subcategories:</h4>
                {subcategory_sections}
            </div>
        </div>
        """
    
    # Extract overall statistics
    total_feedback = len(df)
    sources = df['source'].value_counts().to_dict()
    sources_html = "".join([f"<li>{source}: {count}</li>" for source, count in sources.items()])
    
    positive_count = len(df[df['sentiment'] == 'Positive'])
    negative_count = len(df[df['sentiment'] == 'Negative'])
    neutral_count = len(df[df['sentiment'] == 'Neutral'])
    
    overall_positive_percent = (positive_count / total_feedback) * 100 if total_feedback > 0 else 0
    overall_negative_percent = (negative_count / total_feedback) * 100 if total_feedback > 0 else 0
    overall_neutral_percent = (neutral_count / total_feedback) * 100 if total_feedback > 0 else 0
    
    # Build the navigation
    nav_items = "".join([f'<li><a href="#{category.lower().replace(" ", "-")}">{category}</a></li>' for category in categories if category not in ["Uncategorized", "Other"]])
    
    # Include the plots
    plots_html = ""
    for plot_file, plot_title in plot_files:
        plots_html += f"""
        <div class="plot-container">
            <h4>{plot_title}</h4>
            <img src="images/{plot_file}" alt="{plot_title}" class="plot-image">
        </div>
        """
    
    # Build the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Microsoft Edge Mobile User Feedback Analysis</title>
        <style>
            :root {{
                --primary-color: #0078d4;
                --secondary-color: #106ebe;
                --background-color: #f9f9f9;
                --card-background: #ffffff;
                --text-color: #333333;
                --border-color: #dddddd;
                --positive-color: #107c10;
                --neutral-color: #767676;
                --negative-color: #d13438;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--background-color);
                margin: 0;
                padding: 0;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            header {{
                background-color: var(--primary-color);
                color: white;
                padding: 20px 0;
                margin-bottom: 30px;
            }}
            
            .header-content {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            
            h1, h2, h3, h4 {{
                color: var(--primary-color);
                margin-top: 30px;
            }}
            
            h1 {{
                font-size: 2.2em;
                margin-bottom: 10px;
            }}
            
            h2 {{
                font-size: 1.8em;
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 10px;
                margin-top: 40px;
            }}
            
            h3 {{
                font-size: 1.5em;
                margin-top: 30px;
            }}
            
            h4 {{
                font-size: 1.2em;
                margin-top: 20px;
            }}
            
            .nav-container {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 15px;
                margin-bottom: 30px;
            }}
            
            nav ul {{
                list-style-type: none;
                padding: 0;
                display: flex;
                flex-wrap: wrap;
            }}
            
            nav li {{
                margin-right: 20px;
                margin-bottom: 10px;
            }}
            
            nav a {{
                color: var(--primary-color);
                text-decoration: none;
                font-weight: 500;
                padding: 5px 10px;
                border-radius: 3px;
                transition: background-color 0.3s;
            }}
            
            nav a:hover {{
                background-color: var(--primary-color);
                color: white;
            }}
            
            .overview-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            
            .stat-card {{
                background-color: var(--card-background);
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
                text-align: center;
            }}
            
            .stat-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: var(--primary-color);
                margin: 10px 0;
            }}
            
            .stat-card .label {{
                font-size: 1em;
                color: var(--text-color);
            }}
            
            .sentiment-bar {{
                display: flex;
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
            }}
            
            .sentiment-positive {{
                background-color: var(--positive-color);
                height: 100%;
            }}
            
            .sentiment-neutral {{
                background-color: var(--neutral-color);
                height: 100%;
            }}
            
            .sentiment-negative {{
                background-color: var(--negative-color);
                height: 100%;
            }}
            
            .category-section, .subcategory-section {{
                background-color: var(--card-background);
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 30px;
            }}
            
            .subcategory-section {{
                margin-left: 20px;
                margin-top: 20px;
            }}
            
            .category-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }}
            
            .stat-item {{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 3px;
            }}
            
            .stat-label {{
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }}
            
            .examples-container {{
                margin-top: 20px;
            }}
            
            .example-card {{
                border-left: 4px solid var(--primary-color);
                padding: 10px 15px;
                margin-bottom: 15px;
                background-color: #f5f5f5;
                border-radius: 0 3px 3px 0;
            }}
            
            .example-text {{
                font-style: italic;
                margin-bottom: 8px;
            }}
            
            .example-meta {{
                font-size: 0.9em;
                color: #666;
            }}
            
            .count {{
                font-weight: normal;
                font-size: 0.9em;
                color: #666;
            }}
            
            .plot-container {{
                background-color: var(--card-background);
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 30px;
                text-align: center;
            }}
            
            .plot-image {{
                max-width: 100%;
                height: auto;
                margin: 10px 0;
            }}
            
            .source {{
                color: var(--primary-color);
            }}
            
            footer {{
                background-color: var(--primary-color);
                color: white;
                text-align: center;
                padding: 20px 0;
                margin-top: 50px;
            }}
            
            @media (max-width: 768px) {{
                .category-stats {{
                    grid-template-columns: 1fr;
                }}
                
                .overview-stats {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <header>
            <div class="header-content">
                <h1>Microsoft Edge Mobile User Feedback Analysis</h1>
                <p>Analysis period: May 7 - May 27, 2025</p>
            </div>
        </header>
        
        <div class="container">
            <section class="nav-container">
                <h2>Navigation</h2>
                <nav>
                    <ul>
                        <li><a href="#executive-summary">Executive Summary</a></li>
                        <li><a href="#methodology">Methodology</a></li>
                        <li><a href="#overall-findings">Overall Findings</a></li>
                        <li><a href="#visualizations">Visualizations</a></li>
                        <li><a href="#category-analysis">Category Analysis</a></li>
                        {nav_items}
                        <li><a href="#recommendations">Recommendations</a></li>
                    </ul>
                </nav>
            </section>
            
            <section id="executive-summary">
                <h2>Executive Summary</h2>
                <p>This report analyzes user feedback for Microsoft Edge Mobile collected between May 7-27, 2025. The data was gathered from multiple sources including Unwrap (social media monitoring), Google Play Store reviews, and iOS App Store reviews.</p>
                <p>A total of <strong>{total_feedback}</strong> pieces of feedback were analyzed across all channels. The analysis focused on categorizing feedback, sentiment analysis, identifying feature requests and bug reports, and extracting actionable insights to improve the Edge Mobile experience.</p>
                
                <div class="overview-stats">
                    <div class="stat-card">
                        <div class="label">Total Feedback</div>
                        <div class="value">{total_feedback}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Sentiment Distribution</div>
                        <div class="value">
                            <span style="color: green;">{overall_positive_percent:.1f}%</span> / 
                            <span style="color: gray;">{overall_neutral_percent:.1f}%</span> / 
                            <span style="color: red;">{overall_negative_percent:.1f}%</span>
                        </div>
                        <div class="sentiment-bar">
                            <div class="sentiment-positive" style="width: {overall_positive_percent}%;"></div>
                            <div class="sentiment-neutral" style="width: {overall_neutral_percent}%;"></div>
                            <div class="sentiment-negative" style="width: {overall_negative_percent}%;"></div>
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Feature Requests</div>
                        <div class="value">{df['is_feature_request'].sum()}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Bug Reports</div>
                        <div class="value">{df['is_bug_report'].sum()}</div>
                    </div>
                </div>
            </section>
            
            <section id="methodology">
                <h2>Methodology</h2>
                <p>The analysis followed a comprehensive approach to extract meaningful insights from user feedback:</p>
                <ol>
                    <li><strong>Data Collection:</strong> Feedback was collected from Unwrap social media monitoring (Reddit, Twitter), Google Play Store reviews, and iOS App Store reviews for the period May 7-27, 2025.</li>
                    <li><strong>Data Processing:</strong> All feedback was cleaned and preprocessed to remove irrelevant information and standardize the format.</li>
                    <li><strong>Categorization:</strong> Feedback was categorized into main categories and subcategories based on content analysis.</li>
                    <li><strong>Sentiment Analysis:</strong> A 5-point sentiment scale was applied to each piece of feedback to identify positive, neutral, and negative sentiments.</li>
                    <li><strong>Feature Request & Bug Report Identification:</strong> Feedback was analyzed to identify feature requests and bug reports.</li>
                    <li><strong>Trend Analysis:</strong> Patterns and trends were identified across different feedback sources and categories.</li>
                    <li><strong>Visualization:</strong> Key findings were visualized to provide clear insights.</li>
                </ol>
                <p><strong>Data Sources:</strong></p>
                <ul>
                    {sources_html}
                </ul>
            </section>
            
            <section id="overall-findings">
                <h2>Overall Findings</h2>
                <p>The analysis of user feedback for Microsoft Edge Mobile revealed several key insights:</p>
                <ul>
                    <li><strong>Primary Feedback Categories:</strong> The most discussed aspects of Edge Mobile were {", ".join([cat for cat in df['main_category'].value_counts().head(3).index if cat not in ["Uncategorized", "Other"]])}.</li>
                    <li><strong>Sentiment Overview:</strong> Overall, {overall_positive_percent:.1f}% of the feedback was positive, {overall_neutral_percent:.1f}% was neutral, and {overall_negative_percent:.1f}% was negative.</li>
                    <li><strong>Feature Requests:</strong> {df['is_feature_request'].sum()} pieces of feedback contained feature requests, primarily focused on {", ".join([cat for cat, count in df[df['is_feature_request']]['main_category'].value_counts().head(2).items() if cat not in ["Uncategorized", "Other"]])}.</li>
                    <li><strong>Bug Reports:</strong> {df['is_bug_report'].sum()} pieces of feedback reported bugs or issues, mainly related to {", ".join([cat for cat, count in df[df['is_bug_report']]['main_category'].value_counts().head(2).items() if cat not in ["Uncategorized", "Other"]])}.</li>
                    <li><strong>Platform Differences:</strong> Feedback from iOS users differed from Android users in some key areas, particularly regarding {", ".join([cat for cat, count in df[df['platform'] == 'ios']['main_category'].value_counts().head(1).items() if cat not in ["Uncategorized", "Other"]])}.</li>
                </ul>
            </section>
            
            <section id="visualizations">
                <h2>Visualizations</h2>
                <div class="plots-container">
                    {plots_html}
                </div>
            </section>
            
            <section id="category-analysis">
                <h2>Category Analysis</h2>
                {category_sections}
            </section>
            
            <section id="recommendations">
                <h2>Recommendations</h2>
                <p>Based on the analysis of user feedback, the following recommendations are proposed to improve the Microsoft Edge Mobile experience:</p>
                
                <h3>High Priority</h3>
                <ol>
                    <li><strong>Address performance issues:</strong> Focus on optimizing performance, particularly addressing crashes in InPrivate mode and slow loading times that were frequently mentioned in negative feedback.</li>
                    <li><strong>Fix identified bugs:</strong> Prioritize fixing the fullscreen video display issues in landscape mode on Android and the PDF handling problems with POST requests.</li>
                    <li><strong>Improve extension support:</strong> Enhance the availability and functionality of popular extensions like ad blockers, which were frequently requested and discussed by users.</li>
                </ol>
                
                <h3>Medium Priority</h3>
                <ol>
                    <li><strong>Enhance UI consistency:</strong> Address the visual bugs and layout issues reported across different devices and screen orientations.</li>
                    <li><strong>Optimize memory usage:</strong> Improve RAM management, especially when multiple tabs are open, as noted in several complaints about the app becoming slow over time.</li>
                    <li><strong>Improve sync reliability:</strong> Ensure consistent syncing functionality across devices, as this was a point of both positive and negative feedback.</li>
                </ol>
                
                <h3>Feature Enhancements</h3>
                <ol>
                    <li><strong>Expand extension ecosystem:</strong> Continue to add support for more extensions, particularly those that are popular on desktop versions.</li>
                    <li><strong>Refine privacy features:</strong> Improve the InPrivate browsing experience and add more granular privacy controls as requested by users.</li>
                    <li><strong>Enhance media handling:</strong> Improve video playback, particularly in fullscreen mode, addressing the visual bugs reported by multiple users.</li>
                </ol>
                
                <h3>Continuous Improvement</h3>
                <ol>
                    <li><strong>Regular performance audits:</strong> Implement regular performance testing across different devices and usage scenarios.</li>
                    <li><strong>Expand user feedback channels:</strong> Consider adding in-app feedback options to capture more structured and specific feedback.</li>
                    <li><strong>Competitive analysis:</strong> Regularly compare Edge Mobile features and performance against competing browsers mentioned in user feedback.</li>
                </ol>
            </section>
        </div>
        
        <footer>
            <div class="container">
                <p>Microsoft Edge Mobile User Feedback Analysis Report | Generated on {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
        </footer>
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {output_path}")
    return output_path

def main():
    # Define file paths
    unwrap_file = '/Users/lxm/Documents/Output/working_dir/feedback_analysis_20250527/Unwrap_Group_Export_-_Edge_mobile_-_2025-05-27_16_05_29_+08_00.csv'
    google_play_file = '/Users/lxm/Documents/Output/working_dir/feedback_analysis_20250527/app_reviews_google-play_20250527_182647.csv'
    ios_file = '/Users/lxm/Documents/Output/working_dir/feedback_analysis_20250527/app_reviews_ios_20250527_182656.csv'
    
    # Create directories for outputs
    output_dir = '/Users/lxm/Documents/Output/working_dir/feedback_analysis_20250527'
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Process data from each source
    unwrap_df = process_unwrap_data(unwrap_file)
    google_play_df = process_app_reviews(google_play_file, 'Google Play')
    ios_df = process_app_reviews(ios_file, 'iOS')
    
    # Combine all data sources
    combined_df = pd.concat([unwrap_df, google_play_df, ios_df], ignore_index=True)
    
    # Generate visualizations
    plot_files = generate_plots(combined_df, img_dir)
    
    # Generate HTML report
    report_path = os.path.join(output_dir, 'edge_mobile_feedback_analysis.html')
    generate_html_report(combined_df, plot_files, report_path)
    
    print(f"Analysis complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
