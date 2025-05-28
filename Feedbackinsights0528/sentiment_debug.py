from analyze_feedback import analyze_sentiment

test_text = "Delete the Download history is a pain It is a pain to use MS Edge. There is no option to clear the long list in Downloads. The only option is click the three dots, then delete them one at a time. Can not believe software giant like Microsoft makes app so dumb like this."

result = analyze_sentiment(test_text)
print(f"Sentiment: {result[0]}, Score: {result[1]}")

# 手动分析文本中的正面和负面词汇
text = test_text.lower()

positive_words = ['great', 'good', 'awesome', 'excellent', 'amazing', 'love', 'best', 'fantastic', 
                 'perfect', 'helpful', 'easy', 'impressed', 'smooth', 'useful', 'happy', 'like',
                 'recommended', 'convenient', 'improved', 'fast', 'works well']

negative_words = ['bad', 'terrible', 'horrible', 'awful', 'slow', 'bug', 'crash', 'problem', 'issue',
                 'disappointing', 'hate', 'difficult', 'broken', 'fail', 'error', 'poor', 'useless',
                 'waste', 'annoying', 'frustrating', 'worst', 'unusable', 'cannot', 'not working', 'pain', 'dumb']

pos_matches = [word for word in positive_words if word in text]
neg_matches = [word for word in negative_words if word in text]

pos_score = len(pos_matches)
neg_score = len(neg_matches)

print(f"Positive words found ({pos_score}): {pos_matches}")
print(f"Negative words found ({neg_score}): {neg_matches}")
print(f"Raw scores - Positive: {pos_score}, Negative: {neg_score}")

# 检查否定词
negations = ['not', 'no ', "don't", "doesn't", "didn't", "isn't", "aren't", "can't", "couldn't", 'never']
negated_pos = []

for neg in negations:
    for pos in positive_words:
        pattern1 = f"{neg} {pos}"
        pattern2 = f"{neg}{pos}"
        if pattern1 in text:
            negated_pos.append(pattern1)
        if pattern2 in text:
            negated_pos.append(pattern2)

print(f"Negated positive words: {negated_pos}")

# 打印文本中每个词，用于检查
words = text.split()
print("\nAll words in text:")
print(words)
