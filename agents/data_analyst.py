"""
Data Analyst Agent for the Corporate Intelligence Agentic System.

This agent specializes in statistical analysis, pattern recognition,
and extracting insights from collected data to support decision-making.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from loguru import logger
from langchain.schema import HumanMessage

from core.config import get_config
from core.models import (
    AgentType, AgentResult, AnalysisStatus, DataPoint, StatisticalAnalysis
)
from core.utils import (
    get_llm, create_system_message, extract_keywords, clean_text,
    create_summary_statistics, detect_outliers, retry_with_backoff,
    generate_cache_key, get_db_manager
)


class DataAnalyst:
    """Data Analyst Agent for statistical analysis and pattern recognition."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm()
        self.db_manager = get_db_manager()
        self.agent_type = AgentType.DATA_ANALYST
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    async def analyze_data(self, data_points: List[DataPoint], topic: str) -> StatisticalAnalysis:
        """
        Perform comprehensive statistical analysis on collected data.
        
        Args:
            data_points: List of DataPoint objects to analyze
            topic: The topic being analyzed
            
        Returns:
            StatisticalAnalysis object with comprehensive results
        """
        logger.info(f"Starting data analysis for topic: {topic} with {len(data_points)} data points")
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = generate_cache_key("data_analysis", topic, len(data_points))
            cached_result = self.db_manager.get_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached data analysis result for: {topic}")
                return StatisticalAnalysis(**cached_result)
            
            # Convert data points to DataFrame for analysis
            df = self._create_dataframe(data_points)
            
            # Perform various analyses
            correlation_matrix = await self._analyze_correlations(df)
            trend_analysis = await self._analyze_trends(df, topic)
            outlier_detection = await self._detect_outliers(df)
            clustering_results = await self._perform_clustering(df)
            time_series_analysis = await self._analyze_time_series(df)
            summary_statistics = await self._calculate_summary_statistics(df)
            
            # Create comprehensive analysis result
            analysis = StatisticalAnalysis(
                correlation_matrix=correlation_matrix,
                trend_analysis=trend_analysis,
                outlier_detection=outlier_detection,
                clustering_results=clustering_results,
                time_series_analysis=time_series_analysis,
                summary_statistics=summary_statistics
            )
            
            # Cache the result
            self.db_manager.set_cache(cache_key, analysis.dict(), ttl_seconds=7200)  # 2 hours
            
            logger.info(f"Completed data analysis for {topic}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in data analysis for {topic}: {e}")
            raise e
    
    def _create_dataframe(self, data_points: List[DataPoint]) -> pd.DataFrame:
        """Convert DataPoint objects to pandas DataFrame."""
        try:
            data = []
            for dp in data_points:
                # Extract text features
                text_length = len(dp.content)
                word_count = len(dp.content.split())
                
                # Extract keywords
                keywords = extract_keywords(dp.content, max_keywords=10)
                keyword_count = len(keywords)
                
                # Extract date features
                if dp.published_date:
                    date = dp.published_date
                    day_of_week = date.weekday()
                    hour = date.hour
                    is_weekend = 1 if day_of_week >= 5 else 0
                else:
                    date = dp.collected_date
                    day_of_week = date.weekday()
                    hour = date.hour
                    is_weekend = 1 if day_of_week >= 5 else 0
                
                data.append({
                    'id': str(dp.id),
                    'source': dp.source.value,
                    'title': dp.title,
                    'content': dp.content,
                    'text_length': text_length,
                    'word_count': word_count,
                    'keyword_count': keyword_count,
                    'sentiment_score': dp.sentiment_score or 0.0,
                    'relevance_score': dp.relevance_score or 0.0,
                    'published_date': dp.published_date,
                    'collected_date': dp.collected_date,
                    'day_of_week': day_of_week,
                    'hour': hour,
                    'is_weekend': is_weekend,
                    'url': dp.url,
                    'keywords': keywords
                })
            
            df = pd.DataFrame(data)
            
            # Add derived features
            df['sentiment_category'] = df['sentiment_score'].apply(self._categorize_sentiment)
            df['relevance_category'] = df['relevance_score'].apply(self._categorize_relevance)
            df['content_complexity'] = df['word_count'] / df['text_length']
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return pd.DataFrame()
    
    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _categorize_relevance(self, score: float) -> str:
        """Categorize relevance score."""
        if score > 0.7:
            return 'high'
        elif score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    async def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between numerical variables."""
        try:
            # Select numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col not in ['day_of_week', 'hour', 'is_weekend']]
            
            if len(numerical_cols) < 2:
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = df[numerical_cols].corr()
            
            # Convert to dictionary format
            correlations = {}
            for col1 in numerical_cols:
                correlations[col1] = {}
                for col2 in numerical_cols:
                    if col1 != col2:
                        correlations[col1][col2] = float(correlation_matrix.loc[col1, col2])
            
            # Find significant correlations
            significant_correlations = {}
            for col1 in numerical_cols:
                significant_correlations[col1] = {}
                for col2 in numerical_cols:
                    if col1 != col2:
                        corr_value = correlation_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.3:  # Only include significant correlations
                            significant_correlations[col1][col2] = float(corr_value)
            
            return significant_correlations
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    async def _analyze_trends(self, df: pd.DataFrame, topic: str) -> Dict[str, Any]:
        """Analyze trends in the data."""
        try:
            trends = {}
            
            # Time-based trends
            if 'published_date' in df.columns and df['published_date'].notna().any():
                df_time = df[df['published_date'].notna()].copy()
                df_time['date'] = pd.to_datetime(df_time['published_date']).dt.date
                
                # Daily sentiment trends
                daily_sentiment = df_time.groupby('date')['sentiment_score'].mean()
                trends['daily_sentiment'] = {
                    'mean': float(daily_sentiment.mean()),
                    'trend': 'increasing' if daily_sentiment.iloc[-1] > daily_sentiment.iloc[0] else 'decreasing',
                    'volatility': float(daily_sentiment.std())
                }
                
                # Daily relevance trends
                daily_relevance = df_time.groupby('date')['relevance_score'].mean()
                trends['daily_relevance'] = {
                    'mean': float(daily_relevance.mean()),
                    'trend': 'increasing' if daily_relevance.iloc[-1] > daily_relevance.iloc[0] else 'decreasing',
                    'volatility': float(daily_relevance.std())
                }
            
            # Source-based trends
            source_stats = df.groupby('source').agg({
                'sentiment_score': ['mean', 'count'],
                'relevance_score': ['mean', 'count'],
                'text_length': 'mean'
            }).round(3)
            
            trends['source_analysis'] = source_stats.to_dict()
            
            # Content complexity trends
            trends['content_complexity'] = {
                'mean': float(df['content_complexity'].mean()),
                'std': float(df['content_complexity'].std()),
                'distribution': df['content_complexity'].describe().to_dict()
            }
            
            # Keyword frequency analysis
            all_keywords = []
            for keywords in df['keywords']:
                all_keywords.extend(keywords)
            
            keyword_freq = pd.Series(all_keywords).value_counts()
            trends['top_keywords'] = keyword_freq.head(20).to_dict()
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
    
    async def _detect_outliers(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect outliers in the data."""
        try:
            outliers = []
            
            # Numerical columns for outlier detection
            numerical_cols = ['text_length', 'word_count', 'sentiment_score', 'relevance_score']
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            for col in available_cols:
                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                
                for idx in outlier_indices:
                    outliers.append({
                        'data_point_id': df.loc[idx, 'id'],
                        'column': col,
                        'value': float(df.loc[idx, col]),
                        'method': 'IQR',
                        'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
                        'title': df.loc[idx, 'title'][:100] + '...' if len(df.loc[idx, 'title']) > 100 else df.loc[idx, 'title']
                    })
            
            # Z-score method for extreme outliers
            for col in available_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                extreme_outlier_indices = df[z_scores > 3].index
                
                for idx in extreme_outlier_indices:
                    outliers.append({
                        'data_point_id': df.loc[idx, 'id'],
                        'column': col,
                        'value': float(df.loc[idx, col]),
                        'method': 'Z-score',
                        'z_score': float(z_scores[df.index.get_loc(idx)]),
                        'title': df.loc[idx, 'title'][:100] + '...' if len(df.loc[idx, 'title']) > 100 else df.loc[idx, 'title']
                    })
            
            return outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return []
    
    async def _perform_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on the data."""
        try:
            clustering_results = {}
            
            # Prepare features for clustering
            features = ['text_length', 'word_count', 'sentiment_score', 'relevance_score']
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 2:
                return clustering_results
            
            # Prepare data
            X = df[available_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            n_clusters = min(5, len(X) // 10)  # Adaptive number of clusters
            if n_clusters < 2:
                n_clusters = 2
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to dataframe
            df_clustered = df.copy()
            df_clustered['cluster'] = cluster_labels
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'avg_sentiment': float(cluster_data['sentiment_score'].mean()),
                    'avg_relevance': float(cluster_data['relevance_score'].mean()),
                    'avg_text_length': float(cluster_data['text_length'].mean()),
                    'sources': cluster_data['source'].value_counts().to_dict(),
                    'sample_titles': cluster_data['title'].head(3).tolist()
                }
            
            clustering_results['kmeans'] = {
                'n_clusters': n_clusters,
                'cluster_analysis': cluster_analysis,
                'inertia': float(kmeans.inertia_),
                'silhouette_score': float(self._calculate_silhouette_score(X_scaled, cluster_labels))
            }
            
            # PCA for dimensionality reduction and visualization
            if len(available_features) > 2:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                clustering_results['pca'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'cumulative_variance': float(pca.explained_variance_ratio_.sum())
                }
            
            return clustering_results
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            return {}
    
    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(X, labels)
        except Exception:
            return 0.0
    
    async def _analyze_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series patterns in the data."""
        try:
            time_series_analysis = {}
            
            if 'published_date' not in df.columns or df['published_date'].isna().all():
                return time_series_analysis
            
            # Prepare time series data
            df_time = df[df['published_date'].notna()].copy()
            df_time['date'] = pd.to_datetime(df_time['published_date']).dt.date
            
            # Daily patterns
            daily_patterns = df_time.groupby('date').agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'relevance_score': ['mean', 'std', 'count'],
                'text_length': 'mean'
            })
            
            time_series_analysis['daily_patterns'] = {
                'sentiment': {
                    'mean': float(daily_patterns[('sentiment_score', 'mean')].mean()),
                    'std': float(daily_patterns[('sentiment_score', 'std')].mean()),
                    'trend': self._calculate_trend(daily_patterns[('sentiment_score', 'mean')])
                },
                'relevance': {
                    'mean': float(daily_patterns[('relevance_score', 'mean')].mean()),
                    'std': float(daily_patterns[('relevance_score', 'std')].mean()),
                    'trend': self._calculate_trend(daily_patterns[('relevance_score', 'mean')])
                },
                'volume': {
                    'mean': float(daily_patterns[('sentiment_score', 'count')].mean()),
                    'trend': self._calculate_trend(daily_patterns[('sentiment_score', 'count')])
                }
            }
            
            # Weekly patterns
            df_time['week'] = pd.to_datetime(df_time['published_date']).dt.isocalendar().week
            weekly_patterns = df_time.groupby('week').agg({
                'sentiment_score': 'mean',
                'relevance_score': 'mean',
                'text_length': 'mean'
            })
            
            time_series_analysis['weekly_patterns'] = {
                'sentiment_trend': self._calculate_trend(weekly_patterns['sentiment_score']),
                'relevance_trend': self._calculate_trend(weekly_patterns['relevance_score']),
                'content_length_trend': self._calculate_trend(weekly_patterns['text_length'])
            }
            
            # Autocorrelation analysis
            if len(daily_patterns) > 10:
                sentiment_series = daily_patterns[('sentiment_score', 'mean')]
                relevance_series = daily_patterns[('relevance_score', 'mean')]
                
                time_series_analysis['autocorrelation'] = {
                    'sentiment_lag1': float(sentiment_series.autocorr(lag=1)) if len(sentiment_series) > 1 else 0.0,
                    'relevance_lag1': float(relevance_series.autocorr(lag=1)) if len(relevance_series) > 1 else 0.0
                }
            
            return time_series_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing time series: {e}")
            return {}
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series."""
        if len(series) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        try:
            summary = {}
            
            # Basic statistics
            summary['data_overview'] = {
                'total_records': len(df),
                'unique_sources': df['source'].nunique(),
                'date_range': {
                    'earliest': df['published_date'].min().isoformat() if df['published_date'].notna().any() else None,
                    'latest': df['published_date'].max().isoformat() if df['published_date'].notna().any() else None
                }
            }
            
            # Numerical statistics
            numerical_cols = ['text_length', 'word_count', 'sentiment_score', 'relevance_score']
            for col in numerical_cols:
                if col in df.columns:
                    summary[f'{col}_stats'] = create_summary_statistics(df[col].dropna().tolist())
            
            # Categorical statistics
            summary['source_distribution'] = df['source'].value_counts().to_dict()
            summary['sentiment_distribution'] = df['sentiment_category'].value_counts().to_dict()
            summary['relevance_distribution'] = df['relevance_category'].value_counts().to_dict()
            
            # Content analysis
            summary['content_analysis'] = {
                'avg_text_length': float(df['text_length'].mean()),
                'avg_word_count': float(df['word_count'].mean()),
                'avg_content_complexity': float(df['content_complexity'].mean()),
                'total_words': int(df['word_count'].sum()),
                'unique_keywords': len(set([kw for keywords in df['keywords'] for kw in keywords]))
            }
            
            # Quality metrics
            summary['quality_metrics'] = {
                'avg_sentiment_score': float(df['sentiment_score'].mean()),
                'avg_relevance_score': float(df['relevance_score'].mean()),
                'sentiment_std': float(df['sentiment_score'].std()),
                'relevance_std': float(df['relevance_score'].std()),
                'missing_dates': int(df['published_date'].isna().sum()),
                'data_completeness': float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return {}
    
    def create_agent_result(self, topic: str, analysis: StatisticalAnalysis, start_time: datetime) -> AgentResult:
        """Create an agent result for the data analysis."""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return AgentResult(
            agent_type=self.agent_type,
            status=AnalysisStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output={
                'analysis': analysis.dict(),
                'topic': topic,
                'analysis_types': ['correlation', 'trend', 'outlier', 'clustering', 'time_series', 'summary']
            },
            metadata={
                'outliers_detected': len(analysis.outlier_detection) if analysis.outlier_detection else 0,
                'clusters_identified': len(analysis.clustering_results.get('kmeans', {}).get('cluster_analysis', {})) if analysis.clustering_results else 0,
                'significant_correlations': sum(len(corrs) for corrs in analysis.correlation_matrix.values()) if analysis.correlation_matrix else 0
            }
        ) 