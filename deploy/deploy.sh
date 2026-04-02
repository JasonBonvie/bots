#!/bin/bash
# ============================================================
# Nekalshi Deploy Script
# Run this FROM YOUR MAC to push code to the VPS
# Usage: bash deploy/deploy.sh
# ============================================================

set -e

VPS_IP="31.97.13.238"
VPS_USER="root"
APP_DIR="/opt/nekalshi"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================="
echo "  Deploying Nekalshi to VPS"
echo "  VPS: $VPS_USER@$VPS_IP"
echo "  From: $LOCAL_DIR"
echo "========================================="

# Check SSH connectivity
echo ""
echo "[1/4] Testing SSH connection..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$VPS_USER@$VPS_IP" "echo ok" > /dev/null 2>&1; then
    echo "  ERROR: Cannot connect to VPS. Make sure:"
    echo "    1. VPS is started on Hostinger panel"
    echo "    2. SSH key is configured, or use: ssh-copy-id $VPS_USER@$VPS_IP"
    exit 1
fi
echo "  Connected."

# Create remote directory
echo ""
echo "[2/4] Preparing remote directory..."
ssh "$VPS_USER@$VPS_IP" "mkdir -p $APP_DIR"

# Sync files (exclude stuff we don't need on server)
echo ""
echo "[3/4] Uploading files..."
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude 'node_modules/' \
    --exclude '*.db' \
    --exclude '.DS_Store' \
    "$LOCAL_DIR/" "$VPS_USER@$VPS_IP:$APP_DIR/"

echo "  Files uploaded."

# Upload sensitive files separately (only if they exist and aren't already there)
echo ""
echo "[4/4] Checking sensitive files..."

if [ -f "$LOCAL_DIR/.env" ]; then
    scp "$LOCAL_DIR/.env" "$VPS_USER@$VPS_IP:$APP_DIR/.env"
    echo "  .env uploaded."
else
    echo "  WARNING: No .env file found locally!"
fi

if [ -f "$LOCAL_DIR/kalshi_private_key.pem" ]; then
    scp "$LOCAL_DIR/kalshi_private_key.pem" "$VPS_USER@$VPS_IP:$APP_DIR/kalshi_private_key.pem"
    echo "  kalshi_private_key.pem uploaded."
else
    echo "  WARNING: No kalshi_private_key.pem found locally!"
fi

echo ""
echo "========================================="
echo "  Upload complete!"
echo "========================================="
echo ""
echo "  FIRST TIME? Run setup on the VPS:"
echo "    ssh $VPS_USER@$VPS_IP 'bash $APP_DIR/deploy/setup-vps.sh'"
echo ""
echo "  UPDATING? Just restart the service:"
echo "    ssh $VPS_USER@$VPS_IP 'systemctl restart nekalshi'"
echo ""
