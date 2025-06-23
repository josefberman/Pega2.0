"""
Data models for the Corporate Intelligence Agentic System.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class AnalysisDepth(str, Enum):
    """Analysis depth levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


class AnalysisStatus(str, Enum):
    """Analysis request status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(str, Enum):
    """Agent types in the system."""
    TECH_RESEARCHER = "tech_researcher"
    DATA_MINER = "data_miner"
    DATA_ANALYST = "data_analyst"
    OPERATIONS_RESEARCHER = "operations_researcher"
    ORCHESTRATOR = "orchestrator"


class DataSource(str, Enum):
    """Data source types."""
    NEWS_WEBSITES = "news_websites"
    SOCIAL_MEDIA = "social_media"
    FINANCIAL_DATA = "financial_data"
    GOVERNMENT_DATABASES = "government_databases"
    COMPANY_WEBSITES = "company_websites"
    ACADEMIC_PAPERS = "academic_papers"
    PATENT_DATABASES = "patent_databases"
    INDUSTRY_REPORTS = "industry_reports"
    CONFERENCE_PROCEEDINGS = "conference_proceedings"


class AnalysisRequest(BaseModel):
    """Analysis request model."""
    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(..., description="The topic to analyze")
    depth: AnalysisDepth = Field(default=AnalysisDepth.COMPREHENSIVE)
    description: Optional[str] = Field(None, description="Additional description")
    priority: int = Field(default=1, ge=1, le=10)
    requested_sources: Optional[List[DataSource]] = Field(None)
    custom_parameters: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class AgentResult(BaseModel):
    """Individual agent result model."""
    agent_type: AgentType
    status: AnalysisStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = Field(None, description="Duration in seconds")
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DataPoint(BaseModel):
    """Data point model for collected data."""
    id: UUID = Field(default_factory=uuid4)
    source: DataSource
    url: Optional[str] = None
    title: str
    content: str
    published_date: Optional[datetime] = None
    collected_date: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    relevance_score: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class StatisticalAnalysis(BaseModel):
    """Statistical analysis results."""
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    outlier_detection: Optional[List[Dict[str, Any]]] = None
    clustering_results: Optional[Dict[str, Any]] = None
    time_series_analysis: Optional[Dict[str, Any]] = None
    summary_statistics: Optional[Dict[str, Any]] = None


class ForecastingResult(BaseModel):
    """Forecasting and simulation results."""
    predictions: Optional[Dict[str, List[float]]] = None
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    scenario_analysis: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None


class TechResearchReport(BaseModel):
    """Technology research report."""
    executive_summary: str
    technology_overview: str
    market_analysis: Dict[str, Any]
    key_players: List[Dict[str, Any]]
    trends_and_developments: List[str]
    risks_and_challenges: List[str]
    opportunities: List[str]
    strategic_recommendations: List[str]
    technical_details: Optional[Dict[str, Any]] = None
    references: List[str] = []


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    request_id: UUID
    status: AnalysisStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None
    
    # Agent results
    tech_research_result: Optional[AgentResult] = None
    data_mining_result: Optional[AgentResult] = None
    data_analysis_result: Optional[AgentResult] = None
    operations_research_result: Optional[AgentResult] = None
    
    # Processed data
    collected_data: List[DataPoint] = []
    statistical_analysis: Optional[StatisticalAnalysis] = None
    forecasting_results: Optional[ForecastingResult] = None
    tech_research_report: Optional[TechResearchReport] = None
    
    # Final output
    executive_summary: Optional[str] = None
    key_insights: List[str] = []
    strategic_recommendations: List[str] = []
    risk_assessment: Optional[Dict[str, Any]] = None
    market_forecast: Optional[Dict[str, Any]] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    error_log: List[str] = []
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class WorkflowState(BaseModel):
    """Workflow state for LangGraph."""
    request: AnalysisRequest
    result: Optional[AnalysisResult] = None
    current_step: Optional[str] = None
    completed_steps: List[str] = []
    step_results: Dict[str, Any] = {}
    errors: List[str] = []
    metadata: Dict[str, Any] = {}


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: AgentType
    recipient: AgentType
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=1)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    active_requests: int
    completed_requests: int
    failed_requests: int
    average_response_time: float
    agent_performance: Dict[AgentType, Dict[str, Any]]
    system_health: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CacheEntry(BaseModel):
    """Cache entry for storing results."""
    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Utility functions for model operations
def create_analysis_request(
    topic: str,
    depth: AnalysisDepth = AnalysisDepth.COMPREHENSIVE,
    description: Optional[str] = None,
    priority: int = 1,
    requested_sources: Optional[List[DataSource]] = None,
    custom_parameters: Optional[Dict[str, Any]] = None
) -> AnalysisRequest:
    """Create a new analysis request."""
    return AnalysisRequest(
        topic=topic,
        depth=depth,
        description=description,
        priority=priority,
        requested_sources=requested_sources,
        custom_parameters=custom_parameters
    )


def create_workflow_state(request: AnalysisRequest) -> WorkflowState:
    """Create a new workflow state."""
    return WorkflowState(
        request=request,
        result=AnalysisResult(
            request_id=request.id,
            status=AnalysisStatus.PENDING,
            created_at=datetime.utcnow()
        )
    )


def update_analysis_status(result: AnalysisResult, status: AnalysisStatus) -> AnalysisResult:
    """Update analysis result status."""
    result.status = status
    if status == AnalysisStatus.COMPLETED:
        result.completed_at = datetime.utcnow()
        if result.created_at:
            result.total_duration = (result.completed_at - result.created_at).total_seconds()
    return result 