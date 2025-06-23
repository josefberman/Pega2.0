"""
Orchestrator Agent for the Corporate Intelligence Agentic System.

This agent coordinates the activities of all specialized agents,
ensures proper data flow, and generates comprehensive final reports.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from loguru import logger
from langchain.schema import HumanMessage

from core.config import get_config
from core.models import (
    AgentType, AgentResult, AnalysisStatus, AnalysisRequest, AnalysisResult,
    WorkflowState, DataPoint, StatisticalAnalysis, ForecastingResult, TechResearchReport
)
from core.utils import (
    get_llm, create_system_message, retry_with_backoff,
    generate_cache_key, get_db_manager
)

from .tech_researcher import TechResearcher
from .data_miner import DataMiner
from .data_analyst import DataAnalyst
from .operations_researcher import OperationsResearcher


class Orchestrator:
    """Orchestrator Agent for coordinating the entire analysis workflow."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = get_llm()
        self.db_manager = get_db_manager()
        self.agent_type = AgentType.ORCHESTRATOR
        
        # Initialize other agents
        self.tech_researcher = TechResearcher()
        self.data_miner = DataMiner()
        self.data_analyst = DataAnalyst()
        self.operations_researcher = OperationsResearcher()
    
    async def orchestrate_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Orchestrate the complete analysis workflow.
        
        Args:
            request: Analysis request with topic and parameters
            
        Returns:
            Complete AnalysisResult with all agent outputs
        """
        logger.info(f"Starting orchestration for topic: {request.topic}")
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = generate_cache_key("orchestration", request.topic, request.depth.value)
            cached_result = self.db_manager.get_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached orchestration result for: {request.topic}")
                return AnalysisResult(**cached_result)
            
            # Create workflow state
            workflow_state = WorkflowState(request=request)
            
            # Initialize result
            result = AnalysisResult(
                request_id=request.id,
                status=AnalysisStatus.IN_PROGRESS,
                created_at=start_time
            )
            
            # Execute workflow steps
            await self._execute_workflow(workflow_state, result)
            
            # Generate final synthesis
            await self._synthesize_results(result, request.topic)
            
            # Update final status
            result.status = AnalysisStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            result.total_duration = (result.completed_at - result.created_at).total_seconds()
            
            # Cache the result
            self.db_manager.set_cache(cache_key, result.dict(), ttl_seconds=86400)  # 24 hours
            
            logger.info(f"Completed orchestration for {request.topic}")
            return result
            
        except Exception as e:
            logger.error(f"Error in orchestration for {request.topic}: {e}")
            
            # Create error result
            error_result = AnalysisResult(
                request_id=request.id,
                status=AnalysisStatus.FAILED,
                created_at=start_time,
                completed_at=datetime.utcnow(),
                error_log=[f"Orchestration error: {str(e)}"]
            )
            
            return error_result
    
    async def _execute_workflow(self, workflow_state: WorkflowState, result: AnalysisResult):
        """Execute the analysis workflow step by step."""
        try:
            # Step 1: Technology Research
            await self._execute_tech_research(workflow_state, result)
            
            # Step 2: Data Mining
            await self._execute_data_mining(workflow_state, result)
            
            # Step 3: Data Analysis
            await self._execute_data_analysis(workflow_state, result)
            
            # Step 4: Operations Research
            await self._execute_operations_research(workflow_state, result)
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            result.error_log.append(f"Workflow execution error: {str(e)}")
            raise e
    
    async def _execute_tech_research(self, workflow_state: WorkflowState, result: AnalysisResult):
        """Execute technology research step."""
        try:
            logger.info("Starting technology research step")
            step_start = datetime.utcnow()
            
            # Execute tech research
            tech_report = await self.tech_researcher.research_topic(
                workflow_state.request.topic,
                workflow_state.request.depth.value
            )
            
            # Create agent result
            agent_result = self.tech_researcher.create_agent_result(
                workflow_state.request.topic,
                tech_report,
                step_start
            )
            
            # Update workflow state
            workflow_state.completed_steps.append("tech_research")
            workflow_state.step_results["tech_research"] = tech_report.dict()
            
            # Update analysis result
            result.tech_research_result = agent_result
            result.tech_research_report = tech_report
            
            logger.info("Completed technology research step")
            
        except Exception as e:
            logger.error(f"Error in tech research step: {e}")
            result.error_log.append(f"Tech research error: {str(e)}")
            raise e
    
    async def _execute_data_mining(self, workflow_state: WorkflowState, result: AnalysisResult):
        """Execute data mining step."""
        try:
            logger.info("Starting data mining step")
            step_start = datetime.utcnow()
            
            # Execute data mining
            async with self.data_miner as miner:
                data_points = await miner.mine_data(
                    workflow_state.request.topic,
                    workflow_state.request.requested_sources,
                    max_data_points=1000
                )
            
            # Create agent result
            agent_result = self.data_miner.create_agent_result(
                workflow_state.request.topic,
                data_points,
                step_start
            )
            
            # Update workflow state
            workflow_state.completed_steps.append("data_mining")
            workflow_state.step_results["data_mining"] = {
                "data_points_count": len(data_points),
                "sources_used": list(set(dp.source.value for dp in data_points))
            }
            
            # Update analysis result
            result.data_mining_result = agent_result
            result.collected_data = data_points
            
            logger.info(f"Completed data mining step: {len(data_points)} data points")
            
        except Exception as e:
            logger.error(f"Error in data mining step: {e}")
            result.error_log.append(f"Data mining error: {str(e)}")
            raise e
    
    async def _execute_data_analysis(self, workflow_state: WorkflowState, result: AnalysisResult):
        """Execute data analysis step."""
        try:
            logger.info("Starting data analysis step")
            step_start = datetime.utcnow()
            
            if not result.collected_data:
                logger.warning("No data points available for analysis")
                return
            
            # Execute data analysis
            statistical_analysis = await self.data_analyst.analyze_data(
                result.collected_data,
                workflow_state.request.topic
            )
            
            # Create agent result
            agent_result = self.data_analyst.create_agent_result(
                workflow_state.request.topic,
                statistical_analysis,
                step_start
            )
            
            # Update workflow state
            workflow_state.completed_steps.append("data_analysis")
            workflow_state.step_results["data_analysis"] = {
                "analysis_types": ["correlation", "trend", "outlier", "clustering", "time_series", "summary"]
            }
            
            # Update analysis result
            result.data_analysis_result = agent_result
            result.statistical_analysis = statistical_analysis
            
            logger.info("Completed data analysis step")
            
        except Exception as e:
            logger.error(f"Error in data analysis step: {e}")
            result.error_log.append(f"Data analysis error: {str(e)}")
            raise e
    
    async def _execute_operations_research(self, workflow_state: WorkflowState, result: AnalysisResult):
        """Execute operations research step."""
        try:
            logger.info("Starting operations research step")
            step_start = datetime.utcnow()
            
            if not result.collected_data or not result.statistical_analysis:
                logger.warning("Insufficient data for operations research")
                return
            
            # Execute operations research
            forecasting_result = await self.operations_researcher.perform_operations_research(
                result.collected_data,
                result.statistical_analysis,
                workflow_state.request.topic
            )
            
            # Create agent result
            agent_result = self.operations_researcher.create_agent_result(
                workflow_state.request.topic,
                forecasting_result,
                step_start
            )
            
            # Update workflow state
            workflow_state.completed_steps.append("operations_research")
            workflow_state.step_results["operations_research"] = {
                "analysis_types": ["forecasting", "scenario_analysis", "risk_assessment", "optimization"]
            }
            
            # Update analysis result
            result.operations_research_result = agent_result
            result.forecasting_results = forecasting_result
            
            logger.info("Completed operations research step")
            
        except Exception as e:
            logger.error(f"Error in operations research step: {e}")
            result.error_log.append(f"Operations research error: {str(e)}")
            raise e
    
    async def _synthesize_results(self, result: AnalysisResult, topic: str):
        """Synthesize results from all agents into final insights."""
        try:
            logger.info("Starting results synthesis")
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(result, topic)
            result.executive_summary = executive_summary
            
            # Extract key insights
            key_insights = await self._extract_key_insights(result, topic)
            result.key_insights = key_insights
            
            # Generate strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(result, topic)
            result.strategic_recommendations = strategic_recommendations
            
            # Assess risks
            risk_assessment = await self._assess_overall_risks(result, topic)
            result.risk_assessment = risk_assessment
            
            # Generate market forecast
            market_forecast = await self._generate_market_forecast(result, topic)
            result.market_forecast = market_forecast
            
            logger.info("Completed results synthesis")
            
        except Exception as e:
            logger.error(f"Error in results synthesis: {e}")
            result.error_log.append(f"Results synthesis error: {str(e)}")
    
    async def _generate_executive_summary(self, result: AnalysisResult, topic: str) -> str:
        """Generate executive summary from all agent results."""
        try:
            prompt = f"""
            Generate a comprehensive executive summary for the analysis of: {topic}
            
            Based on the following results:
            
            Tech Research: {result.tech_research_report.executive_summary if result.tech_research_report else 'Not available'}
            
            Data Collection: {len(result.collected_data)} data points collected from various sources
            
            Statistical Analysis: {len(result.statistical_analysis.trend_analysis) if result.statistical_analysis and result.statistical_analysis.trend_analysis else 0} key trends identified
            
            Forecasting: {len(result.forecasting_results.predictions) if result.forecasting_results and result.forecasting_results.predictions else 0} forecasts generated
            
            Create a 2-3 paragraph executive summary that highlights:
            1. Key findings and insights
            2. Market opportunities and risks
            3. Strategic implications for corporate decision-making
            
            Make it concise, actionable, and suitable for executive leadership.
            """
            
            response = await self._call_llm(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return f"Executive summary for {topic} analysis completed with comprehensive insights across technology research, data analysis, and strategic forecasting."
    
    async def _extract_key_insights(self, result: AnalysisResult, topic: str) -> List[str]:
        """Extract key insights from all agent results."""
        try:
            insights = []
            
            # Tech research insights
            if result.tech_research_report:
                insights.extend([
                    f"Technology maturity: {result.tech_research_report.market_analysis.get('maturity', 'Unknown')}",
                    f"Market size: {result.tech_research_report.market_analysis.get('size', 'Unknown')}",
                    f"Key players identified: {len(result.tech_research_report.key_players)}"
                ])
            
            # Data analysis insights
            if result.statistical_analysis and result.statistical_analysis.trend_analysis:
                trend_data = result.statistical_analysis.trend_analysis
                if 'daily_sentiment' in trend_data:
                    insights.append(f"Sentiment trend: {trend_data['daily_sentiment'].get('trend', 'stable')}")
                if 'daily_relevance' in trend_data:
                    insights.append(f"Relevance trend: {trend_data['daily_relevance'].get('trend', 'stable')}")
            
            # Forecasting insights
            if result.forecasting_results and result.forecasting_results.scenario_analysis:
                scenarios = result.forecasting_results.scenario_analysis
                if 'baseline' in scenarios:
                    insights.append(f"Baseline forecast: {scenarios['baseline'].get('sentiment', 0):.2f} sentiment score")
            
            # Risk insights
            if result.forecasting_results and result.forecasting_results.risk_assessment:
                risk_data = result.forecasting_results.risk_assessment
                if 'overall_risk' in risk_data:
                    insights.append(f"Overall risk level: {risk_data['overall_risk'].get('level', 'unknown')}")
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            logger.error(f"Error extracting key insights: {e}")
            return [f"Comprehensive analysis of {topic} completed successfully"]
    
    async def _generate_strategic_recommendations(self, result: AnalysisResult, topic: str) -> List[str]:
        """Generate strategic recommendations from all agent results."""
        try:
            recommendations = []
            
            # Tech research recommendations
            if result.tech_research_report:
                recommendations.extend(result.tech_research_report.strategic_recommendations[:3])
            
            # Data analysis recommendations
            if result.statistical_analysis and result.statistical_analysis.summary_statistics:
                stats = result.statistical_analysis.summary_statistics
                if 'quality_metrics' in stats:
                    quality = stats['quality_metrics']
                    if quality.get('data_completeness', 100) < 80:
                        recommendations.append("Improve data collection processes for better analysis quality")
            
            # Operations research recommendations
            if result.forecasting_results and result.forecasting_results.optimization_results:
                opt_results = result.forecasting_results.optimization_results
                if 'recommendations' in opt_results:
                    recommendations.extend(opt_results['recommendations'])
            
            # Risk-based recommendations
            if result.forecasting_results and result.forecasting_results.risk_assessment:
                risk_data = result.forecasting_results.risk_assessment
                if 'overall_risk' in risk_data and risk_data['overall_risk'].get('recommendations'):
                    recommendations.extend(risk_data['overall_risk']['recommendations'])
            
            return list(set(recommendations))[:10]  # Remove duplicates and limit to 10
            
        except Exception as e:
            logger.error(f"Error generating strategic recommendations: {e}")
            return [
                f"Continue monitoring {topic} trends and developments",
                "Implement regular analysis cycles for strategic planning",
                "Establish cross-functional team for technology assessment"
            ]
    
    async def _assess_overall_risks(self, result: AnalysisResult, topic: str) -> Dict[str, Any]:
        """Assess overall risks from all agent results."""
        try:
            overall_risks = {
                'risk_level': 'medium',
                'risk_factors': [],
                'mitigation_strategies': []
            }
            
            # Combine risk assessments from different agents
            if result.forecasting_results and result.forecasting_results.risk_assessment:
                forecast_risks = result.forecasting_results.risk_assessment
                if 'overall_risk' in forecast_risks:
                    overall_risks['risk_level'] = forecast_risks['overall_risk'].get('level', 'medium')
                    overall_risks['risk_factors'].extend(forecast_risks['overall_risk'].get('recommendations', []))
            
            # Add data quality risks
            if result.statistical_analysis and result.statistical_analysis.summary_statistics:
                stats = result.statistical_analysis.summary_statistics
                if 'quality_metrics' in stats:
                    quality = stats['quality_metrics']
                    if quality.get('data_completeness', 100) < 80:
                        overall_risks['risk_factors'].append("Incomplete data may affect analysis accuracy")
            
            # Add tech research risks
            if result.tech_research_report and result.tech_research_report.risks_and_challenges:
                overall_risks['risk_factors'].extend(result.tech_research_report.risks_and_challenges[:3])
            
            # Generate mitigation strategies
            overall_risks['mitigation_strategies'] = [
                "Implement continuous monitoring and early warning systems",
                "Diversify data sources and analysis methods",
                "Establish regular review cycles for risk assessment",
                "Develop contingency plans for high-risk scenarios"
            ]
            
            return overall_risks
            
        except Exception as e:
            logger.error(f"Error assessing overall risks: {e}")
            return {
                'risk_level': 'medium',
                'risk_factors': [f"Analysis of {topic} completed with standard risk assessment"],
                'mitigation_strategies': ["Continue monitoring and regular assessment"]
            }
    
    async def _generate_market_forecast(self, result: AnalysisResult, topic: str) -> Dict[str, Any]:
        """Generate market forecast from all agent results."""
        try:
            market_forecast = {
                'forecast_period': '12 months',
                'market_outlook': 'stable',
                'growth_projections': {},
                'key_drivers': [],
                'uncertainty_factors': []
            }
            
            # Extract forecasting data
            if result.forecasting_results and result.forecasting_results.predictions:
                predictions = result.forecasting_results.predictions
                if 'sentiment_mean' in predictions:
                    sentiment_forecast = predictions['sentiment_mean']
                    if sentiment_forecast:
                        avg_sentiment = sum(sentiment_forecast) / len(sentiment_forecast)
                        market_forecast['growth_projections']['sentiment'] = {
                            'current': sentiment_forecast[0] if sentiment_forecast else 0,
                            'forecast': avg_sentiment,
                            'trend': 'increasing' if avg_sentiment > sentiment_forecast[0] else 'decreasing'
                        }
            
            # Extract scenario analysis
            if result.forecasting_results and result.forecasting_results.scenario_analysis:
                scenarios = result.forecasting_results.scenario_analysis
                if 'baseline' in scenarios:
                    baseline = scenarios['baseline']
                    market_forecast['market_outlook'] = 'positive' if baseline.get('sentiment', 0) > 0 else 'negative'
            
            # Extract key drivers from tech research
            if result.tech_research_report and result.tech_research_report.trends_and_developments:
                market_forecast['key_drivers'] = result.tech_research_report.trends_and_developments[:5]
            
            # Extract uncertainty factors
            if result.tech_research_report and result.tech_research_report.risks_and_challenges:
                market_forecast['uncertainty_factors'] = result.tech_research_report.risks_and_challenges[:5]
            
            return market_forecast
            
        except Exception as e:
            logger.error(f"Error generating market forecast: {e}")
            return {
                'forecast_period': '12 months',
                'market_outlook': 'stable',
                'growth_projections': {},
                'key_drivers': [f"Analysis of {topic} indicates stable market conditions"],
                'uncertainty_factors': ["Market conditions subject to change based on external factors"]
            }
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with retry logic."""
        def llm_call():
            messages = [
                create_system_message(self.agent_type),
                HumanMessage(content=prompt)
            ]
            return self.llm.predict_messages(messages)
        
        return retry_with_backoff(llm_call, max_retries=3)
    
    def create_agent_result(self, request: AnalysisRequest, result: AnalysisResult, start_time: datetime) -> AgentResult:
        """Create an agent result for the orchestration."""
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        return AgentResult(
            agent_type=self.agent_type,
            status=AnalysisStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output={
                'analysis_result': result.dict(),
                'topic': request.topic,
                'workflow_steps': ['tech_research', 'data_mining', 'data_analysis', 'operations_research']
            },
            metadata={
                'total_duration': duration,
                'agents_executed': 4,
                'data_points_processed': len(result.collected_data) if result.collected_data else 0,
                'analysis_completeness': 'comprehensive' if result.status == AnalysisStatus.COMPLETED else 'partial'
            }
        ) 