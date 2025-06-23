# Ollama Setup Guide

This guide will help you set up Ollama with the Llama 3.2:3B model for the Corporate Intelligence Agentic System.

## Prerequisites

- Windows 10/11, macOS, or Linux
- At least 8GB RAM (16GB recommended)
- 10GB free disk space
- Internet connection for downloading the model

## Installation Steps

### 1. Install Ollama

#### Windows
1. Download Ollama from [https://ollama.ai](https://ollama.ai)
2. Run the installer and follow the setup wizard
3. Ollama will be installed as a Windows service

#### macOS
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Service

#### Windows
Ollama should start automatically as a Windows service. If not:
```cmd
ollama serve
```

#### macOS/Linux
```bash
ollama serve
```

### 3. Pull Llama 3.2:3B Model

```bash
ollama pull llama3.2:3b
```

This will download approximately 2GB of model data. The download time depends on your internet connection.

### 4. Verify Installation

Check if the model is available:
```bash
ollama list
```

You should see `llama3.2:3b` in the list.

### 5. Test the Model

Test basic functionality:
```bash
ollama run llama3.2:3b "Hello, how are you?"
```

## Configuration

The system is already configured to use Ollama with the following settings:

- **API Base URL**: `http://localhost:11434`
- **Model**: `llama3.2:3b`
- **Temperature**: 0.7
- **Max Tokens**: 4000
- **Timeout**: 120 seconds

These settings are in `config.yaml` and can be modified if needed.

## Testing the Integration

Run the test script to verify everything is working:

```bash
python test_ollama.py
```

This will test:
1. Ollama API connection
2. Llama 3.2:3B model generation
3. System integration

## Troubleshooting

### Common Issues

#### 1. Ollama not starting
- **Windows**: Check if Ollama service is running in Services
- **macOS/Linux**: Check if port 11434 is available

#### 2. Model not found
```bash
ollama pull llama3.2:3b
```

#### 3. Out of memory errors
- Close other applications
- Reduce model parameters in `config.yaml`
- Consider using a smaller model

#### 4. Slow response times
- Ensure you have sufficient RAM
- Close unnecessary applications
- Check system resources

### Performance Optimization

#### For Better Performance:
1. **Use SSD storage** for faster model loading
2. **Increase RAM** to 16GB or more
3. **Use GPU acceleration** if available (requires CUDA setup)
4. **Close other applications** to free up resources

#### Model Parameters:
You can adjust these in `config.yaml`:
- `temperature`: Lower values (0.1-0.3) for more focused responses
- `max_tokens`: Reduce for faster responses
- `timeout`: Increase if you experience timeouts

## Alternative Models

If Llama 3.2:3B is too large or slow, you can try:

```bash
# Smaller models
ollama pull llama3.2:1b
ollama pull llama3.2:0.5b

# Or other models
ollama pull mistral:7b
ollama pull codellama:7b
```

Then update the model name in `config.yaml`:
```yaml
llm:
  model: "llama3.2:1b"  # or your chosen model
```

## Security Considerations

- Ollama runs locally, so your data stays on your machine
- No API keys required
- No data sent to external services
- Model weights are stored locally

## Next Steps

Once Ollama is set up and tested:

1. Start the Corporate Intelligence System:
   ```bash
   python -m api.server
   ```

2. Submit your first analysis request:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"topic": "artificial intelligence", "depth": "comprehensive"}'
   ```

## Support

- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Llama 3.2 Documentation](https://llama.meta.com/llama3.2/)

For system-specific issues, check the logs in `./logs/system.log`. 