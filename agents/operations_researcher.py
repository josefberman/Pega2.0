"""
Operations Research Agent for the Corporate Intelligence Agentic System.

This agent specializes in forecasting, simulations, optimization,
and advanced analytics to provide strategic insights for corporations.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from scipy import stats

from loguru import logger
from langchain.schema import HumanMessage

from core.config import get_config
from core.models import (
    AgentType, AgentResult, AnalysisStatus, DataPoint, StatisticalAnalysis, ForecastingResult
)
from core.utils import (
    get_llm, create_system_message, retry_with_backoff,
    generate_cache_key, get_db_manager
)


class OperationsResearcher:
    """Operations Research Agent for forecasting and strategic analysis."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm()
        self.db_manager = get_db_manager()
        self.agent_type = AgentType.OPERATIONS_RESEARCHER
    
    async def perform_operations_research(self, data_points: List[DataPoint], 
                                        statistical_analysis: StatisticalAnalysis,
                                        topic: str) -> ForecastingResult:
        """
        Perform comprehensive operations research analysis.
        
        Args:
            data_points: List of DataPoint objects
            statistical_analysis: Statistical analysis results
            topic: The topic being analyzed
            
        Returns:
            ForecastingResult with comprehensive analysis
        """
        logger.info(f"Starting operations research for topic: {topic}")
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = generate_cache_key("operations_research", topic, len(data_points))
            cached_result = self.db_manager.get_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached operations research result for: {topic}")
                return ForecastingResult(**cached_result)
            
            # Convert data to time series format
            df = self._prepare_time_series_data(data_points)
            
            # Perform various analyses
            predictions = await self._generate_forecasts(df, topic)
            confidence_intervals = await self._calculate_confidence_intervals(df, predictions)
            scenario_analysis = await self._perform_scenario_analysis(df, topic)
            risk_assessment = await self._assess_risks(df, statistical_analysis, topic)
            optimization_results = await self._perform_optimization(df, topic)
            
            # Create comprehensive result
            result = ForecastingResult(
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                scenario_analysis=scenario_analysis,
                risk_assessment=risk_assessment,
                optimization_results=optimization_results
            )
            
            # Cache the result
            self.db_manager.set_cache(cache_key, result.dict(), ttl_seconds=7200)  # 2 hours
            
            logger.info(f"Completed operations research for {topic}")
            return result
            
        except Exception as e:
            logger.error(f"Error in operations research for {topic}: {e}")
            raise e
    
    def _prepare_time_series_data(self, data_points: List[DataPoint]) -> pd.DataFrame:
        """Prepare time series data for analysis."""
        try:
            data = []
            for dp in data_points:
                date = dp.published_date if dp.published_date else dp.collected_date
                
                data.append({
                    'date': date,
                    'sentiment_score': dp.sentiment_score or 0.0,
                    'relevance_score': dp.relevance_score or 0.0,
                    'text_length': len(dp.content),
                    'source': dp.source.value
                })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Create daily aggregations
            daily_data = df.groupby(df['date'].dt.date).agg({
                'sentiment_score': ['mean', 'count'],
                'relevance_score': ['mean', 'count'],
                'text_length': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_data.columns = ['date', 'sentiment_mean', 'sentiment_count',
                                'relevance_mean', 'relevance_count', 'text_length_mean']
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return pd.DataFrame()
    
    async def _generate_forecasts(self, df: pd.DataFrame, topic: str) -> Dict[str, List[float]]:
        """Generate forecasts for various metrics."""
        try:
            forecasts = {}
            
            if len(df) < 5:
                logger.warning("Insufficient data for forecasting")
                return forecasts
            
            # Generate forecasts for different metrics
            metrics = ['sentiment_mean', 'relevance_mean', 'sentiment_count']
            
            for metric in metrics:
                if metric in df.columns:
                    try:
                        ts = df[metric].dropna()
                        if len(ts) > 2:
                            forecast_values = self._simple_forecast(ts)
                            if forecast_values:
                                forecasts[metric] = forecast_values
                    except Exception as e:
                        logger.error(f"Error forecasting {metric}: {e}")
                        continue
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            return {}
    
    def _simple_forecast(self, ts: pd.Series) -> List[float]:
        """Simple linear trend forecast."""
        try:
            x = np.arange(len(ts))
            y = ts.values
            
            # Fit linear regression
            slope, intercept, _, _, _ = stats.linregress(x, y)
            
            # Generate 7-day forecast
            forecast_x = np.arange(len(ts), len(ts) + 7)
            forecast_y = slope * forecast_x + intercept
            
            return forecast_y.tolist()
            
        except Exception as e:
            logger.error(f"Error in simple forecast: {e}")
            return []
    
    async def _calculate_confidence_intervals(self, df: pd.DataFrame, predictions: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Calculate confidence intervals for forecasts."""
        try:
            confidence_intervals = {}
            
            for metric, forecast_values in predictions.items():
                if metric in df.columns and len(forecast_values) > 0:
                    historical_data = df[metric].dropna()
                    if len(historical_data) > 1:
                        volatility = float(historical_data.std())
                        
                        ci_values = []
                        for forecast_value in forecast_values:
                            ci_lower = max(0, forecast_value - 1.96 * volatility)
                            ci_upper = forecast_value + 1.96 * volatility
                            ci_values.append([ci_lower, ci_upper])
                        
                        confidence_intervals[metric] = ci_values
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}
    
    async def _perform_scenario_analysis(self, df: pd.DataFrame, topic: str) -> Dict[str, Any]:
        """Perform scenario analysis for different market conditions."""
        try:
            scenarios = {}
            
            if len(df) < 3:
                return scenarios
            
            # Baseline scenario
            baseline = self._calculate_baseline_scenario(df)
            scenarios['baseline'] = baseline
            
            # Optimistic scenario
            optimistic = {
                'sentiment': baseline['sentiment'] * 1.2,
                'relevance': baseline['relevance'] * 1.2,
                'volume': baseline['volume'] * 1.2
            }
            scenarios['optimistic'] = optimistic
            
            # Pessimistic scenario
            pessimistic = {
                'sentiment': baseline['sentiment'] * 0.8,
                'relevance': baseline['relevance'] * 0.8,
                'volume': baseline['volume'] * 0.8
            }
            scenarios['pessimistic'] = pessimistic
            
            scenarios['probabilities'] = {
                'baseline': 0.6,
                'optimistic': 0.2,
                'pessimistic': 0.2
            }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error performing scenario analysis: {e}")
            return {}
    
    def _calculate_baseline_scenario(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate baseline scenario based on current trends."""
        try:
            sentiment_value = float(df['sentiment_mean'].iloc[-1]) if 'sentiment_mean' in df.columns and len(df) > 0 else 0.0
            relevance_value = float(df['relevance_mean'].iloc[-1]) if 'relevance_mean' in df.columns and len(df) > 0 else 0.0
            volume_value = float(df['sentiment_count'].iloc[-1]) if 'sentiment_count' in df.columns and len(df) > 0 else 0.0
            
            return {
                'sentiment': sentiment_value,
                'relevance': relevance_value,
                'volume': volume_value
            }
            
        except Exception as e:
            logger.error(f"Error calculating baseline scenario: {e}")
            return {'sentiment': 0.0, 'relevance': 0.0, 'volume': 0.0}
    
    async def _assess_risks(self, df: pd.DataFrame, statistical_analysis: StatisticalAnalysis, topic: str) -> Dict[str, Any]:
        """Assess risks based on data analysis."""
        try:
            risk_assessment = {}
            
            # Data quality risks
            risk_assessment['data_quality'] = {
                'insufficient_data': len(df) < 10,
                'data_gaps': len(df) < 20
            }
            
            # Volatility risks
            if 'sentiment_mean' in df.columns:
                sentiment_volatility = float(df['sentiment_mean'].std())
                risk_assessment['volatility'] = {
                    'sentiment_volatility': sentiment_volatility,
                    'high_volatility': sentiment_volatility > 0.3,
                    'risk_level': 'high' if sentiment_volatility > 0.3 else 'medium' if sentiment_volatility > 0.1 else 'low'
                }
            
            # Overall risk score
            risk_score = 0.5  # Default medium risk
            risk_assessment['overall_risk'] = {
                'score': risk_score,
                'level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.3 else 'low',
                'recommendations': ['Continue monitoring trends', 'Collect more data for better analysis']
            }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error assessing risks: {e}")
            return {}
    
    async def _perform_optimization(self, df: pd.DataFrame, topic: str) -> Dict[str, Any]:
        """Perform optimization analysis."""
        try:
            optimization_results = {}
            
            if len(df) < 3:
                return optimization_results
            
            # Simple optimization recommendations
            optimization_results['recommendations'] = [
                'Focus on high-sentiment data sources',
                'Increase data collection frequency',
                'Monitor trend changes weekly'
            ]
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error performing optimization: {e}")
            return {}
    
    def create_agent_result(self, topic: str, result: ForecastingResult, start_time: datetime) -> AgentResult:
        """Create an agent result for the operations research."""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return AgentResult(
            agent_type=self.agent_type,
            status=AnalysisStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output={
                'forecasting_result': result.dict(),
                'topic': topic,
                'analysis_types': ['forecasting', 'scenario_analysis', 'risk_assessment', 'optimization']
            },
            metadata={
                'forecasts_generated': len(result.predictions) if result.predictions else 0,
                'scenarios_analyzed': len(result.scenario_analysis) if result.scenario_analysis else 0,
                'risk_level': result.risk_assessment.get('overall_risk', {}).get('level', 'unknown') if result.risk_assessment else 'unknown'
            }
        ) 