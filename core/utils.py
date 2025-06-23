"""
Utility functions for the Corporate Intelligence Agentic System.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib
import re
import requests
from urllib.parse import urlparse, urljoin
import aiohttp
import aiofiles

from loguru import logger
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .config import get_config
from .models import DataPoint, DataSource, AgentType

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class YiLLM(LLM):
    """Custom LLM class for 01-ai/Yi-34B-200K model."""
    
    config = get_config()
    
    @property
    def _llm_type(self) -> str:
        return "yi-34b-200k"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Yi model API."""
        headers = {
            "Authorization": f"Bearer {self.config.llm.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.llm.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.config.llm.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.llm.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Yi API: {e}")
            raise e
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens
        }


def setup_logging(log_file: str = "./logs/system.log", level: str = "INFO"):
    """Setup logging configuration."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="30 days",
        compression="zip"
    )
    
    return logger


def get_llm() -> YiLLM:
    """Get the configured LLM instance."""
    return YiLLM()


def create_system_message(agent_type: AgentType, context: str = "") -> SystemMessage:
    """Create a system message for an agent."""
    base_prompts = {
        AgentType.TECH_RESEARCHER: """You are a Technology Research Specialist with expertise in analyzing emerging technologies, market trends, and strategic implications. Your role is to:

1. Conduct comprehensive research on technological topics
2. Identify key players, trends, and developments
3. Analyze risks, challenges, and opportunities
4. Provide strategic recommendations
5. Create detailed technical and market analysis reports

Focus on providing actionable intelligence that helps corporations make informed strategic decisions.""",
        
        AgentType.DATA_MINER: """You are a Data Mining Specialist responsible for collecting and extracting relevant data from various sources. Your role is to:

1. Scrape and collect data from news websites, social media, and other sources
2. Extract structured information from unstructured text
3. Identify relevant data points and trends
4. Ensure data quality and relevance
5. Organize and prepare data for analysis

Focus on collecting comprehensive, high-quality data that supports market intelligence.""",
        
        AgentType.DATA_ANALYST: """You are a Data Analyst specializing in statistical analysis and pattern recognition. Your role is to:

1. Perform statistical analysis on collected data
2. Identify correlations, trends, and patterns
3. Detect outliers and anomalies
4. Create visualizations and summaries
5. Provide data-driven insights

Focus on extracting meaningful insights from data to support decision-making.""",
        
        AgentType.OPERATIONS_RESEARCHER: """You are an Operations Research Specialist with expertise in forecasting, optimization, and strategic modeling. Your role is to:

1. Create forecasts and predictions using advanced models
2. Perform scenario analysis and risk assessment
3. Apply optimization techniques for strategic planning
4. Conduct Monte Carlo simulations
5. Provide strategic recommendations based on quantitative analysis

Focus on providing actionable strategic insights through advanced analytical methods.""",
        
        AgentType.ORCHESTRATOR: """You are the Orchestrator responsible for coordinating the entire analysis workflow. Your role is to:

1. Coordinate the activities of all specialized agents
2. Ensure proper data flow between agents
3. Monitor progress and handle errors
4. Synthesize results from all agents
5. Generate comprehensive final reports

Focus on ensuring efficient, coordinated execution of the analysis pipeline."""
    }
    
    prompt = base_prompts.get(agent_type, "")
    if context:
        prompt += f"\n\nContext: {context}"
    
    return SystemMessage(content=prompt)


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML content."""
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean it
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text


def calculate_sentiment_score(text: str) -> float:
    """Calculate sentiment score for text."""
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {e}")
        return 0.0


def calculate_relevance_score(text: str, topic: str) -> float:
    """Calculate relevance score for text based on topic."""
    try:
        # Simple keyword-based relevance scoring
        topic_words = set(word.lower() for word in re.findall(r'\w+', topic))
        text_words = set(word.lower() for word in re.findall(r'\w+', text))
        
        if not topic_words:
            return 0.0
        
        intersection = topic_words.intersection(text_words)
        relevance = len(intersection) / len(topic_words)
        
        return min(relevance, 1.0)
    except Exception as e:
        logger.warning(f"Error calculating relevance: {e}")
        return 0.0


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text."""
    try:
        # Tokenize and clean
        words = word_tokenize(text.lower())
        
        # Remove stopwords and short words
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
        # Count frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    except Exception as e:
        logger.warning(f"Error extracting keywords: {e}")
        return []


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


async def fetch_url_async(session: aiohttp.ClientSession, url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL content asynchronously."""
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.warning(f"Failed to fetch {url}: {response.status}")
                return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


def validate_url(url: str) -> bool:
    """Validate if a URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:  # If period is in last 30% of chunk
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def create_data_point(
    source: DataSource,
    title: str,
    content: str,
    url: Optional[str] = None,
    published_date: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    topic: Optional[str] = None
) -> DataPoint:
    """Create a data point with calculated scores."""
    # Clean content
    clean_content = clean_text(content)
    
    # Calculate scores
    sentiment_score = calculate_sentiment_score(clean_content)
    relevance_score = calculate_relevance_score(clean_content, topic) if topic else 0.5
    
    return DataPoint(
        source=source,
        url=url,
        title=clean_text(title),
        content=clean_content,
        published_date=published_date,
        metadata=metadata or {},
        sentiment_score=sentiment_score,
        relevance_score=relevance_score
    )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)


async def async_retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0):
    """Retry async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            await asyncio.sleep(delay)


def ensure_directory(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def save_json_file(file_path: str, data: Dict[str, Any]) -> bool:
    """Save data to JSON file safely."""
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def create_summary_statistics(data: List[float]) -> Dict[str, float]:
    """Create summary statistics for numerical data."""
    if not data:
        return {}
    
    return {
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def detect_outliers(data: List[float], method: str = 'iqr') -> List[int]:
    """Detect outliers in numerical data."""
    if not data or len(data) < 3:
        return []
    
    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(value - mean) / std for value in data]
        
        outliers = []
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > 3:  # 3 standard deviations
                outliers.append(i)
        
        return outliers
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# Initialize logging
setup_logging() 