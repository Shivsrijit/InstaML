import re
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/nlp", tags=["nlp"])

# Lightweight Lexicon-based Sentiment
POSITIVE_WORDS = {
    'good', 'great', 'awesome', 'excellent', 'happy', 'love', 'wonderful', 'best', 'fantastic',
    'amazing', 'beautiful', 'positive', 'nice', 'cool', 'superb', 'glad', 'enjoy', 'pleased',
    'outstanding', 'terrific', 'perfect', 'exceptional', 'helpful', 'friendly', 'recommend',
    'delighted', 'satisfied', 'positive', 'win', 'valuable', 'success', 'successful', 'clean'
}
NEGATIVE_WORDS = {
    'bad', 'worst', 'terrible', 'awful', 'sad', 'hate', 'poor', 'disappointed', 'angry', 'useless',
    'broken', 'failure', 'fail', 'negative', 'worse', 'horrible', 'boring', 'waste', 'annoying',
    'expensive', 'rude', 'defective', 'slow', 'hate', 'dislike', 'regret', 'error', 'bug', 'crash',
    'wrong', 'unhappy', 'useless', 'difficult', 'pain', 'scam', 'fake', 'cheap'
}

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'for', 'in', 'of',
    'on', 'at', 'by', 'with', 'about', 'as', 'into', 'like', 'through', 'after', 'before',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
    'they', 'them', 'their', 'this', 'that', 'these', 'those', 'am', 'been', 'has', 'have',
    'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may',
    'might', 'must', 'just', 'more', 'most', 'some', 'any', 'all', 'both', 'each', 'few',
    'other', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'who', 'whom', 'which', 'what',
    'when', 'where', 'why', 'how', 'if', 'then', 'else', 'here', 'there'
}

class AnalyzeRequest(BaseModel):
    text: str

def clean_and_tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

def run_nlp_analysis(text: str) -> Dict[str, Any]:
    """Execute statistics, sentiment, keyword, and summary NLP algorithms."""
    text_clean = text.strip()
    if not text_clean:
        raise ValueError("Text content is empty.")

    # 1. Statistics
    words = clean_and_tokenize(text_clean)
    word_count = len(words)
    char_count = len(text_clean)
    
    # Sentence splitting
    sentences = [s.strip() for s in re.split(r'[.!?]+', text_clean) if s.strip()]
    sentence_count = max(1, len(sentences))
    
    avg_word_length = round(sum(len(w) for w in words) / max(1, word_count), 2)
    reading_time_seconds = max(1, int((word_count / 200) * 60))  # 200 WPM average

    # 2. Sentiment Score
    pos_matches = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_matches = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_sentiment_words = pos_matches + neg_matches
    
    if total_sentiment_words > 0:
        sentiment_score = (pos_matches - neg_matches) / total_sentiment_words
    else:
        sentiment_score = 0.0
        
    if sentiment_score > 0.1:
        sentiment = "positive"
    elif sentiment_score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # 3. Keywords/Tags (TF-IDF frequency based)
    freq: Dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
            
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [{"word": k, "count": v} for k, v in sorted_keywords[:12]]

    # 4. Extractive Summarization
    summary = ""
    if len(sentences) <= 3:
        summary = text_clean
    else:
        # Score sentences based on word frequency of non-stop keywords
        sentence_scores = []
        for i, s in enumerate(sentences):
            s_words = clean_and_tokenize(s)
            score = sum(freq.get(w, 0) for w in s_words if w in freq)
            # Normalize by length to prevent bias towards long sentences
            score = score / max(1, len(s_words))
            sentence_scores.append((i, s, score))
            
        # Select top 3 sentences by score
        top_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)[:3]
        # Sort back into original chronological order
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        summary = ". ".join([s[1] for s in top_sentences]) + "."

    return {
        "statistics": {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "reading_time_seconds": reading_time_seconds
        },
        "sentiment": {
            "score": round(sentiment_score, 2),
            "label": sentiment,
            "positive_count": pos_matches,
            "negative_count": neg_matches
        },
        "keywords": keywords,
        "summary": summary
    }

@router.post("/analyze")
def analyze_text(req: AnalyzeRequest):
    """Execute NLP tasks directly on raw text content."""
    try:
        return run_nlp_analysis(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")

@router.post("/analyze-file")
def analyze_text_file(file: UploadFile = File(...)):
    """Upload a plain text notepad file (.txt) and run various NLP analysis tasks."""
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only plain text files (.txt) are supported.")
        
    try:
        content = file.file.read().decode('utf-8', errors='ignore')
        return run_nlp_analysis(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process text file: {str(e)}")
