"""
Configuration management for the Corporate Intelligence Agentic System.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseSettings):
    """LLM configuration settings."""
    provider: str = "ollama"
    model: str = "llama3.2:3b"
    api_base: str = "http://localhost:11434"
    api_key: str = ""  # No API key needed for local Ollama
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 120


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    name: str = "corporate_intelligence"
    user: str = Field(default_factory=lambda: os.getenv("DB_USER", ""))
    password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size: int = 10
    max_overflow: int = 20


class RedisConfig(BaseSettings):
    """Redis configuration settings."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = 0
    max_connections: int = 20


class APIConfig(BaseSettings):
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list = ["*"]
    rate_limit: int = 100


class AgentConfig(BaseSettings):
    """Individual agent configuration."""
    enabled: bool = True
    max_concurrent: int = 3
    timeout: int = 300
    output_format: str = "markdown"


class AgentsConfig(BaseSettings):
    """All agents configuration."""
    tech_researcher: AgentConfig = AgentConfig()
    data_miner: AgentConfig = AgentConfig()
    data_analyst: AgentConfig = AgentConfig()
    operations_researcher: AgentConfig = AgentConfig()
    orchestrator: AgentConfig = AgentConfig()


class DataProcessingConfig(BaseSettings):
    """Data processing configuration."""
    batch_size: int = 1000
    max_file_size: str = "100MB"
    supported_formats: list = ["csv", "json", "xml", "html", "pdf", "excel"]
    storage_type: str = "local"
    storage_path: str = "./data/storage"
    retention_days: int = 365


class ScrapingConfig(BaseSettings):
    """Web scraping configuration."""
    delay_between_requests: float = 1.0
    max_retries: int = 3
    timeout: int = 30
    respect_robots_txt: bool = True
    use_proxies: bool = False
    proxy_list: list = []
    user_agent: str = "CorporateIntelligenceBot/1.0"


class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration."""
    log_level: str = "INFO"
    log_file: str = "./logs/system.log"
    max_log_size: str = "100MB"
    log_retention: int = 30
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    health_check_interval: int = 60


class SecurityConfig(BaseSettings):
    """Security configuration."""
    jwt_secret: str = Field(default_factory=lambda: os.getenv("JWT_SECRET", "default-secret"))
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 100
    burst_size: int = 20


class ExternalAPIsConfig(BaseSettings):
    """External APIs configuration."""
    news_api_key: str = Field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    polygon_api_key: str = Field(default_factory=lambda: os.getenv("POLYGON_API_KEY", ""))
    news_api_base_url: str = "https://newsapi.org/v2"
    financial_api_base_url: str = "https://api.polygon.io"


class WorkflowConfig(BaseSettings):
    """Workflow configuration."""
    default_depth: str = "comprehensive"
    parallel_execution: bool = True
    result_caching: bool = True
    cache_ttl: int = 3600
    workflow_timeout: int = 1800
    retry_attempts: int = 3
    retry_delay: int = 60


class SystemConfig:
    """Main system configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        # Initialize all configuration sections
        self.llm = LLMConfig(**config_data.get('llm', {}))
        self.database = DatabaseConfig(**config_data.get('database', {}))
        self.redis = RedisConfig(**config_data.get('redis', {}))
        self.api = APIConfig(**config_data.get('api', {}))
        self.agents = AgentsConfig(**config_data.get('agents', {}))
        self.data_processing = DataProcessingConfig(**config_data.get('data_processing', {}))
        self.scraping = ScrapingConfig(**config_data.get('scraping', {}))
        self.monitoring = MonitoringConfig(**config_data.get('monitoring', {}))
        self.security = SecurityConfig(**config_data.get('security', {}))
        self.external_apis = ExternalAPIsConfig(**config_data.get('external_apis', {}))
        self.workflow = WorkflowConfig(**config_data.get('workflow', {}))
        
        # Store raw config for backward compatibility
        self.raw_config = config_data
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        if self.database.type == "postgresql":
            return f"postgresql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.name}"
        elif self.database.type == "mysql":
            return f"mysql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.name}"
        elif self.database.type == "sqlite":
            return f"sqlite:///{self.database.name}.db"
        else:
            raise ValueError(f"Unsupported database type: {self.database.type}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        else:
            return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_env_vars = [
            "DB_USER", 
            "DB_PASSWORD",
            "JWT_SECRET"
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"Warning: Missing environment variables: {missing_vars}")
            return False
        
        return True
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload the global configuration."""
    config.reload() 