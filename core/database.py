"""
Database operations for the Corporate Intelligence Agentic System.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
import uuid
import json

from .config import get_config
from .models import (
    AnalysisRequest, AnalysisResult, AgentResult, DataPoint,
    AnalysisStatus, AgentType, DataSource
)

Base = declarative_base()


class AnalysisRequestDB(Base):
    """Database model for analysis requests."""
    __tablename__ = "analysis_requests"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String(500), nullable=False)
    depth = Column(String(20), nullable=False)
    description = Column(Text)
    priority = Column(Integer, default=1)
    requested_sources = Column(JSON)
    custom_parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisResultDB(Base):
    """Database model for analysis results."""
    __tablename__ = "analysis_results"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(36), nullable=False)
    status = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    total_duration = Column(Float)
    tech_research_result = Column(JSON)
    data_mining_result = Column(JSON)
    data_analysis_result = Column(JSON)
    operations_research_result = Column(JSON)
    collected_data = Column(JSON)
    statistical_analysis = Column(JSON)
    forecasting_results = Column(JSON)
    tech_research_report = Column(JSON)
    executive_summary = Column(Text)
    key_insights = Column(JSON)
    strategic_recommendations = Column(JSON)
    risk_assessment = Column(JSON)
    market_forecast = Column(JSON)
    metadata = Column(JSON)
    error_log = Column(JSON)


class DataPointDB(Base):
    """Database model for data points."""
    __tablename__ = "data_points"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String(50), nullable=False)
    url = Column(String(1000))
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    published_date = Column(DateTime)
    collected_date = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    sentiment_score = Column(Float)
    relevance_score = Column(Float)
    request_id = Column(String(36), nullable=False)


class SystemMetricsDB(Base):
    """Database model for system metrics."""
    __tablename__ = "system_metrics"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow)
    active_requests = Column(Integer, default=0)
    completed_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    agent_performance = Column(JSON)
    system_health = Column(JSON)


class CacheEntryDB(Base):
    """Database model for cache entries."""
    __tablename__ = "cache_entries"
    
    key = Column(String(255), primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """Database manager for handling all database operations."""
    
    def __init__(self):
        self.config = get_config()
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        database_url = self.config.get_database_url()
        
        # Create engine with appropriate configuration
        if self.config.database.type == "sqlite":
            self.engine = create_engine(
                database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            self.engine = create_engine(
                database_url,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close a database session."""
        session.close()
    
    # Analysis Request Operations
    def create_analysis_request(self, request: AnalysisRequest) -> str:
        """Create a new analysis request in the database."""
        session = self.get_session()
        try:
            db_request = AnalysisRequestDB(
                id=str(request.id),
                topic=request.topic,
                depth=request.depth.value,
                description=request.description,
                priority=request.priority,
                requested_sources=[source.value for source in request.requested_sources] if request.requested_sources else None,
                custom_parameters=request.custom_parameters,
                created_at=request.created_at,
                updated_at=request.updated_at
            )
            session.add(db_request)
            session.commit()
            return str(request.id)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_analysis_request(self, request_id: str) -> Optional[AnalysisRequest]:
        """Get an analysis request by ID."""
        session = self.get_session()
        try:
            db_request = session.query(AnalysisRequestDB).filter(AnalysisRequestDB.id == request_id).first()
            if not db_request:
                return None
            
            return AnalysisRequest(
                id=uuid.UUID(db_request.id),
                topic=db_request.topic,
                depth=AnalysisDepth(db_request.depth),
                description=db_request.description,
                priority=db_request.priority,
                requested_sources=[DataSource(source) for source in db_request.requested_sources] if db_request.requested_sources else None,
                custom_parameters=db_request.custom_parameters,
                created_at=db_request.created_at,
                updated_at=db_request.updated_at
            )
        finally:
            self.close_session(session)
    
    def update_analysis_request(self, request_id: str, **kwargs) -> bool:
        """Update an analysis request."""
        session = self.get_session()
        try:
            db_request = session.query(AnalysisRequestDB).filter(AnalysisRequestDB.id == request_id).first()
            if not db_request:
                return False
            
            for key, value in kwargs.items():
                if hasattr(db_request, key):
                    setattr(db_request, key, value)
            
            db_request.updated_at = datetime.utcnow()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    # Analysis Result Operations
    def create_analysis_result(self, result: AnalysisResult) -> str:
        """Create a new analysis result in the database."""
        session = self.get_session()
        try:
            db_result = AnalysisResultDB(
                id=str(uuid.uuid4()),
                request_id=str(result.request_id),
                status=result.status.value,
                created_at=result.created_at,
                completed_at=result.completed_at,
                total_duration=result.total_duration,
                tech_research_result=result.tech_research_result.dict() if result.tech_research_result else None,
                data_mining_result=result.data_mining_result.dict() if result.data_mining_result else None,
                data_analysis_result=result.data_analysis_result.dict() if result.data_analysis_result else None,
                operations_research_result=result.operations_research_result.dict() if result.operations_research_result else None,
                collected_data=[data.dict() for data in result.collected_data] if result.collected_data else None,
                statistical_analysis=result.statistical_analysis.dict() if result.statistical_analysis else None,
                forecasting_results=result.forecasting_results.dict() if result.forecasting_results else None,
                tech_research_report=result.tech_research_report.dict() if result.tech_research_report else None,
                executive_summary=result.executive_summary,
                key_insights=result.key_insights,
                strategic_recommendations=result.strategic_recommendations,
                risk_assessment=result.risk_assessment,
                market_forecast=result.market_forecast,
                metadata=result.metadata,
                error_log=result.error_log
            )
            session.add(db_result)
            session.commit()
            return db_result.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_analysis_result(self, request_id: str) -> Optional[AnalysisResult]:
        """Get an analysis result by request ID."""
        session = self.get_session()
        try:
            db_result = session.query(AnalysisResultDB).filter(AnalysisResultDB.request_id == request_id).first()
            if not db_result:
                return None
            
            # Reconstruct the AnalysisResult object
            result = AnalysisResult(
                request_id=uuid.UUID(db_result.request_id),
                status=AnalysisStatus(db_result.status),
                created_at=db_result.created_at,
                completed_at=db_result.completed_at,
                total_duration=db_result.total_duration,
                executive_summary=db_result.executive_summary,
                key_insights=db_result.key_insights or [],
                strategic_recommendations=db_result.strategic_recommendations or [],
                risk_assessment=db_result.risk_assessment,
                market_forecast=db_result.market_forecast,
                metadata=db_result.metadata,
                error_log=db_result.error_log or []
            )
            
            # Reconstruct agent results
            if db_result.tech_research_result:
                result.tech_research_result = AgentResult(**db_result.tech_research_result)
            if db_result.data_mining_result:
                result.data_mining_result = AgentResult(**db_result.data_mining_result)
            if db_result.data_analysis_result:
                result.data_analysis_result = AgentResult(**db_result.data_analysis_result)
            if db_result.operations_research_result:
                result.operations_research_result = AgentResult(**db_result.operations_research_result)
            
            # Reconstruct collected data
            if db_result.collected_data:
                result.collected_data = [DataPoint(**data) for data in db_result.collected_data]
            
            # Reconstruct analysis results
            if db_result.statistical_analysis:
                from .models import StatisticalAnalysis
                result.statistical_analysis = StatisticalAnalysis(**db_result.statistical_analysis)
            if db_result.forecasting_results:
                from .models import ForecastingResult
                result.forecasting_results = ForecastingResult(**db_result.forecasting_results)
            if db_result.tech_research_report:
                from .models import TechResearchReport
                result.tech_research_report = TechResearchReport(**db_result.tech_research_report)
            
            return result
        finally:
            self.close_session(session)
    
    def update_analysis_result(self, request_id: str, result: AnalysisResult) -> bool:
        """Update an analysis result."""
        session = self.get_session()
        try:
            db_result = session.query(AnalysisResultDB).filter(AnalysisResultDB.request_id == request_id).first()
            if not db_result:
                return False
            
            # Update fields
            db_result.status = result.status.value
            db_result.completed_at = result.completed_at
            db_result.total_duration = result.total_duration
            db_result.tech_research_result = result.tech_research_result.dict() if result.tech_research_result else None
            db_result.data_mining_result = result.data_mining_result.dict() if result.data_mining_result else None
            db_result.data_analysis_result = result.data_analysis_result.dict() if result.data_analysis_result else None
            db_result.operations_research_result = result.operations_research_result.dict() if result.operations_research_result else None
            db_result.collected_data = [data.dict() for data in result.collected_data] if result.collected_data else None
            db_result.statistical_analysis = result.statistical_analysis.dict() if result.statistical_analysis else None
            db_result.forecasting_results = result.forecasting_results.dict() if result.forecasting_results else None
            db_result.tech_research_report = result.tech_research_report.dict() if result.tech_research_report else None
            db_result.executive_summary = result.executive_summary
            db_result.key_insights = result.key_insights
            db_result.strategic_recommendations = result.strategic_recommendations
            db_result.risk_assessment = result.risk_assessment
            db_result.market_forecast = result.market_forecast
            db_result.metadata = result.metadata
            db_result.error_log = result.error_log
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    # Data Point Operations
    def store_data_points(self, request_id: str, data_points: List[DataPoint]) -> List[str]:
        """Store multiple data points."""
        session = self.get_session()
        try:
            db_data_points = []
            for data_point in data_points:
                db_data_point = DataPointDB(
                    id=str(data_point.id),
                    source=data_point.source.value,
                    url=data_point.url,
                    title=data_point.title,
                    content=data_point.content,
                    published_date=data_point.published_date,
                    collected_date=data_point.collected_date,
                    metadata=data_point.metadata,
                    sentiment_score=data_point.sentiment_score,
                    relevance_score=data_point.relevance_score,
                    request_id=request_id
                )
                db_data_points.append(db_data_point)
            
            session.add_all(db_data_points)
            session.commit()
            return [str(dp.id) for dp in data_points]
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_data_points(self, request_id: str, limit: Optional[int] = None) -> List[DataPoint]:
        """Get data points for a request."""
        session = self.get_session()
        try:
            query = session.query(DataPointDB).filter(DataPointDB.request_id == request_id)
            if limit:
                query = query.limit(limit)
            
            db_data_points = query.all()
            return [
                DataPoint(
                    id=uuid.UUID(dp.id),
                    source=DataSource(dp.source),
                    url=dp.url,
                    title=dp.title,
                    content=dp.content,
                    published_date=dp.published_date,
                    collected_date=dp.collected_date,
                    metadata=dp.metadata,
                    sentiment_score=dp.sentiment_score,
                    relevance_score=dp.relevance_score
                )
                for dp in db_data_points
            ]
        finally:
            self.close_session(session)
    
    # Cache Operations
    def set_cache(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set a cache entry."""
        session = self.get_session()
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            cache_entry = CacheEntryDB(
                key=key,
                value=json.dumps(value),
                expires_at=expires_at
            )
            session.merge(cache_entry)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cache entry."""
        session = self.get_session()
        try:
            cache_entry = session.query(CacheEntryDB).filter(
                CacheEntryDB.key == key,
                CacheEntryDB.expires_at > datetime.utcnow()
            ).first()
            
            if not cache_entry:
                return None
            
            # Update access count and last accessed
            cache_entry.access_count += 1
            cache_entry.last_accessed = datetime.utcnow()
            session.commit()
            
            return json.loads(cache_entry.value)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def delete_cache(self, key: str) -> bool:
        """Delete a cache entry."""
        session = self.get_session()
        try:
            cache_entry = session.query(CacheEntryDB).filter(CacheEntryDB.key == key).first()
            if cache_entry:
                session.delete(cache_entry)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        session = self.get_session()
        try:
            expired_entries = session.query(CacheEntryDB).filter(
                CacheEntryDB.expires_at <= datetime.utcnow()
            ).all()
            
            count = len(expired_entries)
            for entry in expired_entries:
                session.delete(entry)
            
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.close_session(session)


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager 