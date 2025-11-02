#!/usr/bin/env python3
"""
Debug script to test Gemini API configuration
"""

import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ .env file loaded")
    else:
        print("❌ .env file not found")
except ImportError:
    print("⚠️ python-dotenv not installed")

# Check what's in the environment
print(f"GEMINI_API_KEY in env: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
print(f"GOOGLE_API_KEY in env: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")

# Try different environment variable names
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print(f"Found GEMINI_API_KEY: {api_key[:10]}...")
    
    # Try setting different variable names that the client might expect
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GENAI_API_KEY'] = api_key
    os.environ['API_KEY'] = api_key
    
    print("Set multiple environment variable names for testing...")

# Test with explicit API key passing
try:
    from google import genai
    
    # Method 1: Try with client and environment variables
    print("\n🧪 Testing Method 1: Environment variables...")
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents="Test"
        )
        print("✅ Method 1 worked!")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try with explicit API key (if supported)
    print("\n🧪 Testing Method 2: Explicit API key...")
    try:
        # Check if client accepts api_key parameter
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents="Test"
        )
        print("✅ Method 2 worked!")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    print("\n🔍 Debug info:")
    print(f"Available environment variables starting with 'G':")
    for key in os.environ:
        if key.startswith('G') and 'API' in key:
            print(f"  {key}: {os.environ[key][:10]}...")

except ImportError:
    print("❌ google-generativeai not installed")
except Exception as e:
    print(f"❌ Import error: {e}")

print("\n💡 Solutions to try:")
print("1. Check if API key is valid at: https://makersuite.google.com/app/apikey")
print("2. Try setting: export GOOGLE_API_KEY='your-key' (instead of GEMINI_API_KEY)")
print("3. Update the google-generativeai package: pip install --upgrade google-generativeai")
print("4. Check API quota and billing at: https://console.cloud.google.com/")