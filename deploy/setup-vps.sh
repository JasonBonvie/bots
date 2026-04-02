#!/bin/bash
# ============================================================
# Nekalshi VPS Setup Script
# Run this ON THE VPS after uploading files with deploy.sh
# Usage: bash /opt/nekalshi/deploy/setup-vps.sh
# ============================================================

set -e

APP_DIR="/opt/nekalshi"
DEPLOY_DIR="$APP_DIR/deploy"

echo "========================================="
echo "  Nekalshi VPS Setup"
echo "========================================="

# 1. System updates and dependencies
echo ""
echo "[1/6] Installing system dependencies..."
apt update -qq
apt install -y -qq python3 python3-pip python3-venv nginx ufw > /dev/null 2>&1
echo "  Done."

# 2. Python virtual environment
echo ""
echo "[2/6] Setting up Python environment..."
cd "$APP_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Created virtual environment."
else
    echo "  Virtual environment already exists."
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Dependencies installed."

# 3. Verify critical files
echo ""
echo "[3/6] Checking configuration..."

if [ ! -f "$APP_DIR/.env" ]; then
    echo "  WARNING: .env file not found! Copy it manually:"
    echo "  scp .env root@31.97.13.238:/opt/nekalshi/.env"
fi

if [ ! -f "$APP_DIR/kalshi_private_key.pem" ]; then
    echo "  WARNING: kalshi_private_key.pem not found! Copy it manually:"
    echo "  scp kalshi_private_key.pem root@31.97.13.238:/opt/nekalshi/"
fi

# Secure the private key
if [ -f "$APP_DIR/kalshi_private_key.pem" ]; then
    chmod 600 "$APP_DIR/kalshi_private_key.pem"
    echo "  Private key permissions secured."
fi

if [ -f "$APP_DIR/.env" ]; then
    chmod 600 "$APP_DIR/.env"
    echo "  .env permissions secured."
fi

# 4. Install systemd service
echo ""
echo "[4/6] Installing systemd service..."
cp "$DEPLOY_DIR/nekalshi.service" /etc/systemd/system/nekalshi.service
systemctl daemon-reload
systemctl enable nekalshi
echo "  Service installed and enabled."

# 5. Install nginx config
echo ""
echo "[5/6] Configuring nginx..."
cp "$DEPLOY_DIR/nginx.conf" /etc/nginx/sites-available/nekalshi
ln -sf /etc/nginx/sites-available/nekalshi /etc/nginx/sites-enabled/nekalshi
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
if nginx -t 2>/dev/null; then
    systemctl restart nginx
    echo "  Nginx configured and restarted."
else
    echo "  ERROR: Nginx config test failed!"
    nginx -t
    exit 1
fi

# 6. Firewall
echo ""
echo "[6/6] Configuring firewall..."
ufw allow 22/tcp > /dev/null 2>&1   # SSH
ufw allow 80/tcp > /dev/null 2>&1   # HTTP
ufw allow 443/tcp > /dev/null 2>&1  # HTTPS
ufw --force enable > /dev/null 2>&1
echo "  Firewall configured (ports 22, 80, 443)."

# Start the application
echo ""
echo "========================================="
echo "  Starting Nekalshi..."
echo "========================================="
systemctl restart nekalshi
sleep 3

if systemctl is-active --quiet nekalshi; then
    echo ""
    echo "  Nekalshi is RUNNING!"
    echo ""
    echo "  Dashboard: http://31.97.13.238"
    echo "  Logs:      journalctl -u nekalshi -f"
    echo "  Status:    systemctl status nekalshi"
    echo "  Restart:   systemctl restart nekalshi"
    echo ""
else
    echo ""
    echo "  ERROR: Nekalshi failed to start!"
    echo "  Check logs: journalctl -u nekalshi --no-pager -n 30"
    echo ""
    journalctl -u nekalshi --no-pager -n 15
fi
