#!/usr/bin/env python3
"""
Kalshi API Authentication Diagnostic Script

Run this to test if your Kalshi API credentials are working correctly.
Usage: python test_kalshi_auth.py
"""

import os
import sys
import asyncio
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("KALSHI API AUTHENTICATION DIAGNOSTIC")
print("=" * 60)
print()

# Step 1: Check environment variables
print("1️⃣  CHECKING ENVIRONMENT VARIABLES")
print("-" * 40)

api_key_id = os.getenv("KALSHI_API_KEY_ID", "")
private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")
use_demo = os.getenv("KALSHI_USE_DEMO", "false").lower() == "true"
live_trading = os.getenv("KALSHI_LIVE_TRADING", "false").lower() == "true"

print(f"KALSHI_API_KEY_ID:      {'✅ Set' if api_key_id else '❌ NOT SET'}")
if api_key_id:
    print(f"                        Value: {api_key_id[:8]}...{api_key_id[-4:] if len(api_key_id) > 12 else ''}")

print(f"KALSHI_PRIVATE_KEY_PATH: {private_key_path}")
print(f"KALSHI_USE_DEMO:         {use_demo} ({'Demo API' if use_demo else 'Production API'})")
print(f"KALSHI_LIVE_TRADING:     {live_trading}")
print()

if not api_key_id:
    print("❌ ERROR: KALSHI_API_KEY_ID is not set in your .env file")
    print()
    print("To fix:")
    print("  1. Go to Kalshi → Settings → API Keys")
    print("  2. Create a new API key")
    print("  3. Add to your .env file:")
    print("     KALSHI_API_KEY_ID=your-api-key-id-here")
    sys.exit(1)

# Step 2: Check private key file
print("2️⃣  CHECKING PRIVATE KEY FILE")
print("-" * 40)

key_path = Path(private_key_path)
if not key_path.is_absolute():
    key_path = Path(__file__).parent / private_key_path

print(f"Looking for key at: {key_path}")

if not key_path.exists():
    print(f"❌ ERROR: Private key file not found at {key_path}")
    print()
    print("To fix:")
    print("  1. Go to Kalshi → Settings → API Keys")
    print("  2. Generate a new RSA key pair")
    print("  3. Download the private key and save it as:")
    print(f"     {key_path}")
    print("  4. Make sure the file starts with:")
    print("     -----BEGIN RSA PRIVATE KEY-----")
    print("     or")
    print("     -----BEGIN PRIVATE KEY-----")
    sys.exit(1)

print(f"✅ Private key file exists")

# Read and validate key format
with open(key_path, "rb") as f:
    key_content = f.read()

key_str = key_content.decode('utf-8', errors='ignore')
if "-----BEGIN" not in key_str:
    print("❌ ERROR: Private key file doesn't look like a PEM file")
    print("   File should start with -----BEGIN RSA PRIVATE KEY----- or -----BEGIN PRIVATE KEY-----")
    sys.exit(1)

if "RSA PRIVATE KEY" in key_str:
    print("✅ Key format: RSA PRIVATE KEY (PKCS#1)")
elif "PRIVATE KEY" in key_str:
    print("✅ Key format: PRIVATE KEY (PKCS#8)")
else:
    print("⚠️  Unknown key format")

print(f"✅ Key file size: {len(key_content)} bytes")
print()

# Step 3: Try to load the key
print("3️⃣  LOADING PRIVATE KEY")
print("-" * 40)

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend

    private_key = serialization.load_pem_private_key(
        key_content,
        password=None,
        backend=default_backend()
    )
    print("✅ Private key loaded successfully")

    # Get key info
    key_size = private_key.key_size
    print(f"✅ Key size: {key_size} bits")

    if key_size < 2048:
        print("⚠️  WARNING: Key size is less than 2048 bits, may be rejected")

except Exception as e:
    print(f"❌ ERROR: Failed to load private key: {e}")
    print()
    print("To fix:")
    print("  - Make sure the key file is a valid PEM-encoded RSA private key")
    print("  - Regenerate the key pair on Kalshi")
    sys.exit(1)

print()

# Step 4: Test signature generation (using RSA-PSS like official SDK)
print("4️⃣  TESTING SIGNATURE GENERATION (RSA-PSS)")
print("-" * 40)

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    timestamp = str(int(time.time() * 1000))
    method = "GET"
    path = "/trade-api/v2/portfolio/balance"
    message = f"{timestamp}{method}{path}"

    # Use RSA-PSS padding (matching official Kalshi SDK)
    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )

    signature_b64 = base64.b64encode(signature).decode('utf-8')
    print("✅ RSA-PSS signature generated successfully")
    print(f"   Test message: {message[:50]}...")
    print(f"   Signature: {signature_b64[:40]}...")

except Exception as e:
    print(f"❌ ERROR: Failed to generate signature: {e}")
    sys.exit(1)

print()

# Step 5: Test API connection
print("5️⃣  TESTING KALSHI API CONNECTION")
print("-" * 40)

async def test_api():
    import aiohttp

    base_url = "https://demo-api.elections.kalshi.com" if use_demo else "https://api.elections.kalshi.com"
    print(f"Using API: {base_url}")

    # Test 1: Public endpoint (no auth)
    print()
    print("Testing public endpoint (no auth required)...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/trade-api/v2/events", params={"limit": 1}) as resp:
                if resp.status == 200:
                    print(f"✅ Public API accessible (status: {resp.status})")
                else:
                    text = await resp.text()
                    print(f"⚠️  Public API returned status {resp.status}: {text[:100]}")
    except Exception as e:
        print(f"❌ Cannot reach Kalshi API: {e}")
        return False

    # Test 2: Authenticated endpoint with RSA-PSS
    print()
    print("Testing authenticated endpoint (RSA-PSS signature)...")
    try:
        timestamp = str(int(time.time() * 1000))
        method = "GET"
        path = "/trade-api/v2/portfolio/balance"
        message = f"{timestamp}{method}{path}"

        # Use RSA-PSS padding (matching official Kalshi SDK)
        signature = private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        headers = {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json"
        }

        print(f"   API Key: {api_key_id[:8]}...")
        print(f"   Timestamp: {timestamp}")
        print(f"   Signature (RSA-PSS): {signature_b64[:30]}...")

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}{path}"
            async with session.get(url, headers=headers) as resp:
                text = await resp.text()

                if resp.status == 200:
                    import json
                    data = json.loads(text)
                    balance = data.get("balance", 0) / 100
                    print(f"✅ AUTHENTICATION SUCCESSFUL!")
                    print(f"   Account balance: ${balance:.2f}")
                    return True
                elif resp.status == 401:
                    print(f"❌ AUTHENTICATION FAILED (401)")
                    print(f"   Response: {text[:200]}")
                    print()
                    print("Possible causes:")
                    print("  1. API Key ID doesn't match the private key")
                    print("  2. Private key is for a different Kalshi account")
                    print("  3. Using production key with demo API (or vice versa)")
                    print("  4. API key has been revoked or expired")
                    print()
                    print("To fix:")
                    print("  1. Go to Kalshi → Settings → API Keys")
                    print("  2. Delete existing keys and create a new one")
                    print("  3. Download the new private key")
                    print("  4. Update KALSHI_API_KEY_ID in .env")
                    print("  5. Save the private key to the correct location")
                    if use_demo:
                        print()
                        print("⚠️  You're using DEMO API - make sure your credentials")
                        print("   are from demo.kalshi.com, not the main site!")
                    return False
                else:
                    print(f"❌ Unexpected response (status: {resp.status})")
                    print(f"   Response: {text[:200]}")
                    return False

    except Exception as e:
        print(f"❌ API request failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the async test
success = asyncio.run(test_api())

print()
print("=" * 60)
if success:
    print("✅ ALL TESTS PASSED - Your Kalshi API credentials are working!")
    print()
    print("You can now enable live trading by setting:")
    print("   KALSHI_LIVE_TRADING=true")
    print()
    print("in your .env file and restarting the server.")
else:
    print("❌ AUTHENTICATION FAILED - Please fix the issues above")
print("=" * 60)

sys.exit(0 if success else 1)
