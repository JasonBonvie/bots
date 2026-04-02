#!/bin/bash
# ============================================================
# Quick update script - push code changes and restart
# Run FROM YOUR MAC: bash deploy/update.sh
# ============================================================

set -e

VPS_IP="31.97.13.238"
VPS_USER="root"
APP_DIR="/opt/nekalshi"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Pushing code changes to VPS..."

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv/' \
    --exclude '.venv/' \
    --exclude 'node_modules/' \
    --exclude '*.db' \
    --exclude '.DS_Store' \
    --exclude '.env' \
    --exclude 'kalshi_private_key.pem' \
    "$LOCAL_DIR/" "$VPS_USER@$VPS_IP:$APP_DIR/"

echo ""
echo "Restarting service..."
ssh "$VPS_USER@$VPS_IP" "systemctl restart nekalshi"

sleep 2
ssh "$VPS_USER@$VPS_IP" "systemctl is-active nekalshi && echo 'Nekalshi is running!' || echo 'ERROR: Failed to start'"
echo ""
echo "Dashboard: http://$VPS_IP"
