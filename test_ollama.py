#!/usr/bin/env python3
"""
Test script to verify Ollama connection and Llama 3.2:3B model.
"""

import requests
import json
from core.utils import get_llm, setup_logging

def test_ollama_connection():
    """Test basic Ollama API connection."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("✅ Ollama API is running")
            print(f"Available models: {[model['name'] for model in models.get('models', [])]}")
            return True
        else:
            print(f"❌ Ollama API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama API. Make sure Ollama is running on localhost:11434")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_llama_model():
    """Test Llama 3.2:3B model generation."""
    try:
        # Test data
        test_prompt = "Write a brief summary of artificial intelligence in 2 sentences."
        
        data = {
            "model": "llama3.2:3b",
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "")
            print("✅ Llama 3.2:3B model is working")
            print(f"Test prompt: {test_prompt}")
            print(f"Generated response: {generated_text[:200]}...")
            return True
        else:
            print(f"❌ Model generation failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Llama model: {e}")
        return False

def test_system_integration():
    """Test the system's LLM integration."""
    try:
        setup_logging()
        llm = get_llm()
        
        test_prompt = "What is the main advantage of using local LLMs?"
        response = llm._call(test_prompt)
        
        print("✅ System LLM integration is working")
        print(f"Test prompt: {test_prompt}")
        print(f"System response: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Error testing system integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Ollama and Llama 3.2:3B Integration")
    print("=" * 50)
    
    # Test 1: Ollama API connection
    print("\n1. Testing Ollama API connection...")
    connection_ok = test_ollama_connection()
    
    # Test 2: Llama model generation
    print("\n2. Testing Llama 3.2:3B model...")
    model_ok = test_llama_model()
    
    # Test 3: System integration
    print("\n3. Testing system integration...")
    system_ok = test_system_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Ollama API Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"Llama Model Generation: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"System Integration: {'✅ PASS' if system_ok else '❌ FAIL'}")
    
    if all([connection_ok, model_ok, system_ok]):
        print("\n🎉 All tests passed! Your Ollama setup is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check your Ollama installation and configuration.")
        
        if not connection_ok:
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure Ollama is installed: https://ollama.ai")
            print("2. Start Ollama service: ollama serve")
            print("3. Check if Ollama is running on localhost:11434")
            
        if not model_ok:
            print("\n💡 Model troubleshooting:")
            print("1. Pull the model: ollama pull llama3.2:3b")
            print("2. Check available models: ollama list")
            print("3. Verify model is downloaded correctly")

if __name__ == "__main__":
    main() 