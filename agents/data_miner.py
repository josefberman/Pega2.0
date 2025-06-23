"""
Data Mining Agent for the Corporate Intelligence Agentic System.

This agent specializes in scraping and collecting data from various sources
to provide comprehensive datasets for trend analysis and market intelligence.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from urllib.parse import urljoin, urlparse
import aiohttp
import aiofiles

from loguru import logger
from bs4 import BeautifulSoup
import feedparser
import requests
from newspaper import Article
import pandas as pd

from core.config import get_config
from core.models import (
    AgentType, AgentResult, AnalysisStatus, DataPoint, DataSource
)
from core.utils import (
    get_llm, create_system_message, extract_text_from_html,
    clean_text, calculate_sentiment_score, calculate_relevance_score,
    validate_url, fetch_url_async, create_data_point,
    retry_with_backoff, generate_cache_key, get_db_manager
)


class DataMiner:
    """Data Mining Agent for collecting data from various sources."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm()
        self.db_manager = get_db_manager()
        self.agent_type = AgentType.DATA_MINER
        
        # Initialize session for async requests
        self.session = None
        self.visited_urls: Set[str] = set()
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': self.config.scraping.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def mine_data(self, topic: str, sources: Optional[List[DataSource]] = None, 
                       max_data_points: int = 1000) -> List[DataPoint]:
        """
        Mine data from various sources for a given topic.
        
        Args:
            topic: The topic to mine data for
            sources: List of data sources to use
            max_data_points: Maximum number of data points to collect
            
        Returns:
            List of DataPoint objects
        """
        logger.info(f"Starting data mining for topic: {topic}")
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = generate_cache_key("data_mining", topic, sources, max_data_points)
            cached_result = self.db_manager.get_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached data mining result for: {topic}")
                return [DataPoint(**dp) for dp in cached_result]
            
            # Use default sources if none specified
            if not sources:
                sources = [
                    DataSource.NEWS_WEBSITES,
                    DataSource.SOCIAL_MEDIA,
                    DataSource.FINANCIAL_DATA,
                    DataSource.COMPANY_WEBSITES
                ]
            
            # Collect data from different sources
            all_data_points = []
            
            for source in sources:
                try:
                    source_data = await self._collect_from_source(topic, source, max_data_points // len(sources))
                    all_data_points.extend(source_data)
                    logger.info(f"Collected {len(source_data)} data points from {source.value}")
                except Exception as e:
                    logger.error(f"Error collecting from {source.value}: {e}")
                    continue
            
            # Filter and rank data points
            filtered_data = await self._filter_and_rank_data(all_data_points, topic, max_data_points)
            
            # Cache the result
            self.db_manager.set_cache(cache_key, [dp.dict() for dp in filtered_data], ttl_seconds=3600)  # 1 hour
            
            logger.info(f"Completed data mining for {topic}: {len(filtered_data)} data points")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error in data mining for {topic}: {e}")
            raise e
    
    async def _collect_from_source(self, topic: str, source: DataSource, max_points: int) -> List[DataPoint]:
        """Collect data from a specific source."""
        if source == DataSource.NEWS_WEBSITES:
            return await self._collect_news_data(topic, max_points)
        elif source == DataSource.SOCIAL_MEDIA:
            return await self._collect_social_media_data(topic, max_points)
        elif source == DataSource.FINANCIAL_DATA:
            return await self._collect_financial_data(topic, max_points)
        elif source == DataSource.COMPANY_WEBSITES:
            return await self._collect_company_data(topic, max_points)
        elif source == DataSource.GOVERNMENT_DATABASES:
            return await self._collect_government_data(topic, max_points)
        else:
            logger.warning(f"Unknown data source: {source}")
            return []
    
    async def _collect_news_data(self, topic: str, max_points: int) -> List[DataPoint]:
        """Collect data from news websites."""
        data_points = []
        
        # Define news sources
        news_sources = [
            'https://techcrunch.com',
            'https://www.theverge.com',
            'https://www.wired.com',
            'https://www.engadget.com',
            'https://www.zdnet.com',
            'https://www.cnet.com'
        ]
        
        # Search for topic-related articles
        for source_url in news_sources:
            try:
                articles = await self._search_news_source(source_url, topic)
                for article in articles[:max_points // len(news_sources)]:
                    data_point = await self._extract_article_data(article, source_url)
                    if data_point:
                        data_points.append(data_point)
                
                # Rate limiting
                await asyncio.sleep(self.config.scraping.delay_between_requests)
                
            except Exception as e:
                logger.error(f"Error collecting from {source_url}: {e}")
                continue
        
        return data_points
    
    async def _search_news_source(self, source_url: str, topic: str) -> List[str]:
        """Search for articles on a news source."""
        try:
            # This would typically use the site's search API or RSS feed
            # For now, we'll simulate by searching the main page
            
            if self.session:
                html_content = await fetch_url_async(self.session, source_url)
                if html_content:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Find article links (this is a simplified approach)
                    article_links = []
                    for link in soup.find_all('a', href=True):
                        href = link.get('href')
                        if href and any(keyword in href.lower() for keyword in topic.lower().split()):
                            full_url = urljoin(source_url, href)
                            if validate_url(full_url) and full_url not in self.visited_urls:
                                article_links.append(full_url)
                                self.visited_urls.add(full_url)
                    
                    return article_links[:10]  # Limit to 10 articles per source
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching {source_url}: {e}")
            return []
    
    async def _extract_article_data(self, article_url: str, source_url: str) -> Optional[DataPoint]:
        """Extract data from an article URL."""
        try:
            if self.session:
                html_content = await fetch_url_async(self.session, article_url)
                if not html_content:
                    return None
                
                # Use newspaper3k for article extraction
                article = Article(article_url)
                article.download(input_html=html_content)
                article.parse()
                
                if not article.title or not article.text:
                    return None
                
                # Create data point
                data_point = create_data_point(
                    source=DataSource.NEWS_WEBSITES,
                    title=article.title,
                    content=article.text,
                    url=article_url,
                    published_date=article.publish_date,
                    metadata={
                        'source_url': source_url,
                        'authors': article.authors,
                        'keywords': article.keywords,
                        'summary': article.summary
                    },
                    topic=topic
                )
                
                return data_point
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting article data from {article_url}: {e}")
            return None
    
    async def _collect_social_media_data(self, topic: str, max_points: int) -> List[DataPoint]:
        """Collect data from social media platforms."""
        data_points = []
        
        # This would integrate with social media APIs (Twitter, LinkedIn, etc.)
        # For now, we'll simulate social media data collection
        
        try:
            # Simulate social media posts
            social_posts = await self._simulate_social_media_search(topic, max_points)
            
            for post in social_posts:
                data_point = create_data_point(
                    source=DataSource.SOCIAL_MEDIA,
                    title=post['title'],
                    content=post['content'],
                    url=post.get('url'),
                    published_date=post.get('published_date'),
                    metadata={
                        'platform': post.get('platform'),
                        'author': post.get('author'),
                        'engagement': post.get('engagement', {})
                    },
                    topic=topic
                )
                data_points.append(data_point)
            
        except Exception as e:
            logger.error(f"Error collecting social media data: {e}")
        
        return data_points
    
    async def _simulate_social_media_search(self, topic: str, max_points: int) -> List[Dict[str, Any]]:
        """Simulate social media search results."""
        # This would be replaced with actual API calls
        return [
            {
                'title': f'Social media post about {topic}',
                'content': f'Interesting discussion about {topic} and its implications...',
                'url': f'https://twitter.com/user/status/123456',
                'published_date': datetime.utcnow() - timedelta(hours=i),
                'platform': 'Twitter',
                'author': f'User{i}',
                'engagement': {'likes': 100 + i, 'shares': 10 + i}
            }
            for i in range(min(max_points, 20))
        ]
    
    async def _collect_financial_data(self, topic: str, max_points: int) -> List[DataPoint]:
        """Collect financial data related to the topic."""
        data_points = []
        
        try:
            # This would integrate with financial APIs (Alpha Vantage, Yahoo Finance, etc.)
            financial_data = await self._fetch_financial_data(topic, max_points)
            
            for data in financial_data:
                data_point = create_data_point(
                    source=DataSource.FINANCIAL_DATA,
                    title=data['title'],
                    content=data['content'],
                    url=data.get('url'),
                    published_date=data.get('published_date'),
                    metadata={
                        'data_type': data.get('data_type'),
                        'ticker': data.get('ticker'),
                        'financial_metrics': data.get('metrics', {})
                    },
                    topic=topic
                )
                data_points.append(data_point)
            
        except Exception as e:
            logger.error(f"Error collecting financial data: {e}")
        
        return data_points
    
    async def _fetch_financial_data(self, topic: str, max_points: int) -> List[Dict[str, Any]]:
        """Fetch financial data from APIs."""
        # This would make actual API calls to financial data providers
        return [
            {
                'title': f'Financial analysis: {topic}',
                'content': f'Financial performance and market analysis for {topic}...',
                'url': f'https://finance.yahoo.com/quote/{topic}',
                'published_date': datetime.utcnow() - timedelta(days=i),
                'data_type': 'market_analysis',
                'ticker': topic.upper(),
                'metrics': {'price': 100 + i, 'volume': 1000000 + i * 100000}
            }
            for i in range(min(max_points, 10))
        ]
    
    async def _collect_company_data(self, topic: str, max_points: int) -> List[DataPoint]:
        """Collect data from company websites."""
        data_points = []
        
        try:
            # This would search company websites for relevant information
            company_data = await self._search_company_websites(topic, max_points)
            
            for data in company_data:
                data_point = create_data_point(
                    source=DataSource.COMPANY_WEBSITES,
                    title=data['title'],
                    content=data['content'],
                    url=data.get('url'),
                    published_date=data.get('published_date'),
                    metadata={
                        'company': data.get('company'),
                        'page_type': data.get('page_type'),
                        'company_info': data.get('company_info', {})
                    },
                    topic=topic
                )
                data_points.append(data_point)
            
        except Exception as e:
            logger.error(f"Error collecting company data: {e}")
        
        return data_points
    
    async def _search_company_websites(self, topic: str, max_points: int) -> List[Dict[str, Any]]:
        """Search company websites for relevant information."""
        # This would crawl company websites for relevant content
        return [
            {
                'title': f'Company information about {topic}',
                'content': f'Company {i} provides solutions related to {topic}...',
                'url': f'https://company{i}.com/{topic}',
                'published_date': datetime.utcnow() - timedelta(days=i),
                'company': f'Company {i}',
                'page_type': 'product_page',
                'company_info': {'industry': 'Technology', 'size': 'Large'}
            }
            for i in range(min(max_points, 15))
        ]
    
    async def _collect_government_data(self, topic: str, max_points: int) -> List[DataPoint]:
        """Collect data from government databases."""
        data_points = []
        
        try:
            # This would integrate with government APIs and databases
            government_data = await self._fetch_government_data(topic, max_points)
            
            for data in government_data:
                data_point = create_data_point(
                    source=DataSource.GOVERNMENT_DATABASES,
                    title=data['title'],
                    content=data['content'],
                    url=data.get('url'),
                    published_date=data.get('published_date'),
                    metadata={
                        'agency': data.get('agency'),
                        'document_type': data.get('document_type'),
                        'government_info': data.get('government_info', {})
                    },
                    topic=topic
                )
                data_points.append(data_point)
            
        except Exception as e:
            logger.error(f"Error collecting government data: {e}")
        
        return data_points
    
    async def _fetch_government_data(self, topic: str, max_points: int) -> List[Dict[str, Any]]:
        """Fetch data from government databases."""
        # This would make actual API calls to government databases
        return [
            {
                'title': f'Government report on {topic}',
                'content': f'Official government analysis and regulations related to {topic}...',
                'url': f'https://data.gov/{topic}',
                'published_date': datetime.utcnow() - timedelta(days=i*7),
                'agency': f'Agency {i}',
                'document_type': 'report',
                'government_info': {'jurisdiction': 'Federal', 'category': 'Technology'}
            }
            for i in range(min(max_points, 5))
        ]
    
    async def _filter_and_rank_data(self, data_points: List[DataPoint], topic: str, max_points: int) -> List[DataPoint]:
        """Filter and rank data points based on relevance and quality."""
        try:
            # Filter out low-quality data points
            filtered_points = []
            for dp in data_points:
                if (len(dp.content) > 50 and  # Minimum content length
                    dp.relevance_score > 0.1):  # Minimum relevance score
                    filtered_points.append(dp)
            
            # Sort by relevance score (descending)
            filtered_points.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Take top points up to max_points
            return filtered_points[:max_points]
            
        except Exception as e:
            logger.error(f"Error filtering and ranking data: {e}")
            return data_points[:max_points]
    
    def create_agent_result(self, topic: str, data_points: List[DataPoint], start_time: datetime) -> AgentResult:
        """Create an agent result for the data mining."""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        sources_used = list(set(dp.source.value for dp in data_points))
        avg_relevance = sum(dp.relevance_score for dp in data_points) / len(data_points) if data_points else 0
        avg_sentiment = sum(dp.sentiment_score for dp in data_points) / len(data_points) if data_points else 0
        
        return AgentResult(
            agent_type=self.agent_type,
            status=AnalysisStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output={
                'data_points_count': len(data_points),
                'sources_used': sources_used,
                'topic': topic
            },
            metadata={
                'avg_relevance_score': avg_relevance,
                'avg_sentiment_score': avg_sentiment,
                'urls_visited': len(self.visited_urls),
                'data_quality': 'high' if avg_relevance > 0.5 else 'medium'
            }
        ) 