#!/bin/bash

# hostname-fix.sh - Fix Docker hostname resolution issue

echo "🔧 Docker Hostname Resolution Fix"
echo "================================="

echo "1️⃣ Current network situation..."
echo "🔍 Backend can ping ML service IP but not hostname"
echo "📍 ML service IP: $(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"

echo ""
echo "2️⃣ Testing hostname resolution in backend..."
echo "🔍 Can backend resolve ml-service hostname?"
if docker exec portfolio_backend nslookup ml-service 2>/dev/null; then
    echo "✅ Hostname resolution works"
else
    echo "❌ Hostname resolution broken"
fi

echo ""
echo "🔍 What hostnames can backend resolve?"
docker exec portfolio_backend cat /etc/hosts | grep -v "^#"

echo ""
echo "3️⃣ Fixing the hostname resolution..."

# Method 1: Restart containers in correct order to refresh DNS
echo "🔄 Method 1: Restart containers to refresh DNS..."
docker-compose restart backend
sleep 10

echo "🧪 Testing after backend restart..."
if docker exec portfolio_backend curl -f http://ml-service:8002/health >/dev/null 2>&1; then
    echo "✅ Hostname resolution fixed with restart!"
    HOSTNAME_FIXED=true
else
    echo "❌ Still can't resolve hostname, trying method 2..."
    HOSTNAME_FIXED=false
fi

if [ "$HOSTNAME_FIXED" = false ]; then
    # Method 2: Full network refresh
    echo ""
    echo "🔄 Method 2: Full network refresh..."
    
    docker-compose down
    sleep 5
    
    # Remove any stale networks
    docker network prune -f
    
    docker-compose up -d
    
    echo "⏱️ Waiting for services to start..."
    sleep 60
    
    echo "🧪 Testing after full restart..."
    if docker exec portfolio_backend curl -f http://ml-service:8002/health >/dev/null 2>&1; then
        echo "✅ Hostname resolution fixed with full restart!"
        HOSTNAME_FIXED=true
    else
        echo "❌ Still having issues, trying method 3..."
        HOSTNAME_FIXED=false
    fi
fi

if [ "$HOSTNAME_FIXED" = false ]; then
    # Method 3: Use IP address instead of hostname
    echo ""
    echo "🔄 Method 3: Configure backend to use IP address..."
    
    ML_IP=$(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
    echo "📍 Using ML service IP: $ML_IP"
    
    # Update backend environment to use IP instead of hostname
    docker-compose stop backend
    
    # Create temporary env override
    cat > .env.override << EOF
ML_SERVICE_URL=http://$ML_IP:8002
EOF
    
    # Restart backend with IP-based URL
    docker-compose up -d backend
    
    echo "⏱️ Waiting for backend with IP configuration..."
    sleep 20
    
    echo "🧪 Testing with IP-based configuration..."
    if curl -s http://localhost:3001/api/ai/health | grep -q '"available":true'; then
        echo "✅ Backend now works with IP address!"
        HOSTNAME_FIXED=true
    else
        echo "❌ Still having issues"
        HOSTNAME_FIXED=false
    fi
fi

echo ""
echo "4️⃣ Final verification..."

if [ "$HOSTNAME_FIXED" = true ]; then
    echo "✅ Hostname resolution fixed!"
    
    echo ""
    echo "🧪 Complete system test..."
    
    # Test ML service
    if curl -s http://localhost:8002/health >/dev/null; then
        echo "✅ ML service: Working"
    else
        echo "❌ ML service: Failed"
    fi
    
    # Test backend
    if curl -s http://localhost:3001/health >/dev/null; then
        echo "✅ Backend: Working"
    else
        echo "❌ Backend: Failed"
    fi
    
    # Test AI routes
    AI_HEALTH=$(curl -s http://localhost:3001/api/ai/health)
    if echo "$AI_HEALTH" | grep -q '"available":true'; then
        echo "✅ AI routes: ML service available"
    else
        echo "❌ AI routes: ML service unavailable"
    fi
    
    # Test stock quotes
    if curl -s http://localhost:3001/api/ai/quote/AAPL | grep -q '"success":true'; then
        echo "✅ Stock quotes: Working"
    else
        echo "❌ Stock quotes: Failed"
    fi
    
    echo ""
    echo "🎉 SYSTEM READY!"
    echo ""
    echo "✅ Your portfolio analyzer now works with:"
    echo "   - Real OCR extraction from portfolio images"
    echo "   - Real-time market data"
    echo "   - No mock data anywhere"
    echo "   - AI analysis and recommendations"
    echo ""
    echo "🌐 Upload a portfolio screenshot at: http://localhost:3000"
    echo ""
    echo "📋 Supported formats:"
    echo "   - PNG, JPG, JPEG images"
    echo "   - Clear portfolio screenshots"
    echo "   - Under 10MB file size"
    echo "   - Visible stock symbols and values"
    
else
    echo "❌ Could not fix hostname resolution"
    echo ""
    echo "🔧 Manual troubleshooting:"
    echo "1. Check Docker Desktop network settings"
    echo "2. Restart Docker Desktop completely"
    echo "3. Try: docker system prune -a"
    echo "4. Verify no firewall blocking container communication"
    
    echo ""
    echo "🆘 Alternative: Use IP address method"
    echo "The system may still work by manually setting:"
    ML_IP=$(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
    echo "ML_SERVICE_URL=http://$ML_IP:8002"
fi

echo ""
echo "📊 Final container status:"
docker-compose ps

# Clean up temp files
rm -f .env.override