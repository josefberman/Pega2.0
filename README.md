# Corporate Intelligence Agentic System

A comprehensive agentic system built with LangGraph to help corporations analyze market trends, technological shifts, and strategic opportunities.

## System Overview

This system consists of five specialized agents working together to provide comprehensive market intelligence:

1. **Tech Researcher**: Creates comprehensive reports on technological topics, identifying trends, players, risks, and strategic aspects
2. **Data Miner**: Scrapes the internet for large datasets to analyze trends and market shifts
3. **Data Analyst**: Performs statistical analysis, finds correlations, outliers, and insights from mined data
4. **Operations Researcher**: Creates forecasts, deep analysis, simulations using data science methodologies
5. **Orchestrator**: Coordinates and activates agents in the optimal sequence

## Architecture

```
├── agents/                 # Individual agent implementations
│   ├── tech_researcher.py
│   ├── data_miner.py
│   ├── data_analyst.py
│   ├── operations_researcher.py
│   └── orchestrator.py
├── core/                   # Core system components
│   ├── config.py          # Configuration management
│   ├── models.py          # Data models and schemas
│   ├── database.py        # Database operations
│   └── utils.py           # Utility functions
├── data/                   # Data storage and processing
│   ├── scrapers/          # Web scraping modules
│   ├── processors/        # Data processing pipelines
│   └── storage/           # Data storage handlers
├── api/                    # API endpoints
│   ├── routes.py
│   └── server.py
├── workflows/              # LangGraph workflow definitions
│   └── main_workflow.py
├── tests/                  # Test suite
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker configuration
└── config.yaml            # System configuration
```

## Features

- **Multi-Agent Coordination**: LangGraph-based workflow orchestration
- **Advanced LLM Integration**: Uses 01-ai/Yi-34B-200K model for all agents
- **Comprehensive Data Pipeline**: From web scraping to advanced analytics
- **Real-time Market Intelligence**: Continuous monitoring and analysis
- **Strategic Forecasting**: Advanced simulations and predictions
- **Scalable Architecture**: Docker-based deployment with microservices

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the System**:
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your API keys and settings
   ```

3. **Run the System**:
   ```bash
   python -m api.server
   ```

4. **Submit Analysis Request**:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"topic": "artificial intelligence", "depth": "comprehensive"}'
   ```

## Configuration

The system uses a YAML configuration file (`config.yaml`) for:
- API keys and endpoints
- Model parameters
- Database connections
- Agent-specific settings
- Workflow configurations

## API Endpoints

- `POST /analyze` - Submit a new analysis request
- `GET /status/{request_id}` - Check analysis status
- `GET /results/{request_id}` - Retrieve analysis results
- `GET /health` - System health check

## Development

### Running Tests
```bash
pytest tests/
```

### Docker Deployment
```bash
docker-compose up -d
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 