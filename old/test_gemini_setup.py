#!/usr/bin/env python3
"""
Test script to verify .env file and Gemini API setup
"""

import os
from pathlib import Path

# Test .env file loading
try:
    from dotenv import load_dotenv
    print("✅ python-dotenv is installed")
    
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ .env file found and loaded")
    else:
        print("❌ .env file not found")
        exit(1)
        
except ImportError:
    print("❌ python-dotenv not installed")
    print("Install with: pip install python-dotenv")
    exit(1)

# Test API key
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    # Mask the key for security
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"✅ GEMINI_API_KEY loaded: {masked_key}")
else:
    print("❌ GEMINI_API_KEY not found in .env file")
    print("Make sure your .env file contains:")
    print("GEMINI_API_KEY=your-actual-api-key")
    exit(1)

# Test Gemini API
try:
    from google import genai
    print("✅ google-generativeai is installed")
    
    # Set the API key in environment for the client
    os.environ['GOOGLE_API_KEY'] = api_key
    client = genai.Client()
    print("✅ Gemini API client configured successfully")
    
    # Try different model names, starting with more available ones
    model_names = [
        'gemini-1.5-flash',  # Usually most available
        'gemini-1.5-pro',    # Good performance, usually available
        'gemini-2.0-flash-exp',  # Experimental, may be limited
        'gemini-2.5-pro'     # Latest but may be overloaded
    ]
    working_model = None
    
    for model_name in model_names:
        try:
            print(f"\n🧪 Testing model: {model_name}...")
            prompt = "What is agriculture in one sentence?"
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            if response and response.text:
                print(f"✅ API test successful with model: {model_name}!")
                print(f"Response: {response.text}")
                working_model = model_name
                break
            else:
                print(f"⚠️  Model {model_name} returned no response")
                
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg or "overloaded" in error_msg.lower():
                print(f"⚠️  Model {model_name} is overloaded, trying next...")
            elif "404" in error_msg or "not found" in error_msg.lower():
                print(f"⚠️  Model {model_name} not available, trying next...")
            else:
                print(f"⚠️  Model {model_name} failed: {error_msg[:100]}...")
            continue
    
    if not working_model:
        print("❌ No working Gemini models found")
        print("\nThis could be due to:")
        print("1. Models being overloaded (try again later)")
        print("2. API quota limits")
        print("3. Regional availability")
    else:
        print(f"\n✅ Gemini API is ready! Using model: {working_model}")
        
except ImportError:
    print("❌ google-generativeai not installed")
    print("Install with: pip install google-generativeai")
except Exception as e:
    print(f"❌ Gemini API test failed: {e}")
    if "API_KEY" in str(e):
        print("Check that your API key is valid and active")

print("\n🚀 If all tests passed, you can now run:")
print("python src/better_rag.py")