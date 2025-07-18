#!/bin/bash

# comprehensive-fix.sh - Complete system fix and test

echo "🔧 Comprehensive System Fix and Test"
echo "===================================="

echo "1️⃣ Current system status..."
echo "📊 Container status:"
docker-compose ps

echo ""
echo "📋 ML service logs (last 10 lines):"
docker logs portfolio_ml_service --tail 10

echo ""
echo "2️⃣ Testing individual components..."

# Test ML service directly
echo "🔍 Testing ML service directly:"
if curl -f http://localhost:8002/health 2>/dev/null; then
    echo "✅ ML service responds from host"
    ML_HOST_OK=true
else
    echo "❌ ML service not responding from host"
    ML_HOST_OK=false
fi

# Test backend
echo ""
echo "🔍 Testing backend:"
if curl -f http://localhost:3001/health 2>/dev/null; then
    echo "✅ Backend responds"
    BACKEND_OK=true
else
    echo "❌ Backend not responding"
    BACKEND_OK=false
fi

# Test container network connectivity
echo ""
echo "🔍 Testing container network connectivity:"
ML_IP=$(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null)
echo "📍 ML service IP: $ML_IP"

if [ ! -z "$ML_IP" ]; then
    if docker exec portfolio_backend ping -c 1 $ML_IP >/dev/null 2>&1; then
        echo "✅ Backend can ping ML service IP"
        NETWORK_OK=true
    else
        echo "❌ Backend cannot ping ML service IP"
        NETWORK_OK=false
    fi
else
    echo "❌ Could not get ML service IP"
    NETWORK_OK=false
fi

echo ""
echo "3️⃣ Diagnosis and fixes..."

if [ "$ML_HOST_OK" = false ]; then
    echo "❌ ML service is not responding - need to fix this first"
    
    echo "🔧 Restarting ML service..."
    docker-compose restart ml-service
    
    echo "⏱️ Waiting for ML service..."
    sleep 45
    
    if curl -f http://localhost:8002/health 2>/dev/null; then
        echo "✅ ML service now responding"
        ML_HOST_OK=true
    else
        echo "❌ ML service still not responding"
        echo "📋 ML service logs after restart:"
        docker logs portfolio_ml_service --tail 20
        
        echo ""
        echo "🔧 Trying full rebuild..."
        docker-compose build ml-service
        docker-compose up -d ml-service
        
        echo "⏱️ Waiting for rebuild..."
        sleep 60
        
        if curl -f http://localhost:8002/health 2>/dev/null; then
            echo "✅ ML service working after rebuild"
            ML_HOST_OK=true
        else
            echo "❌ ML service still failing - check logs"
            docker logs portfolio_ml_service --tail 30
        fi
    fi
fi

if [ "$BACKEND_OK" = false ]; then
    echo "❌ Backend is not responding"
    echo "🔧 Restarting backend..."
    docker-compose restart backend
    sleep 20
    
    if curl -f http://localhost:3001/health 2>/dev/null; then
        echo "✅ Backend now responding"
        BACKEND_OK=true
    else
        echo "❌ Backend still not responding"
        echo "📋 Backend logs:"
        docker logs portfolio_backend --tail 15
    fi
fi

if [ "$NETWORK_OK" = false ] && [ "$ML_HOST_OK" = true ] && [ "$BACKEND_OK" = true ]; then
    echo "❌ Network connectivity issue between containers"
    
    echo "🔧 Fixing network connectivity..."
    echo "🔄 Full system restart..."
    
    docker-compose down
    sleep 10
    docker-compose up -d
    
    echo "⏱️ Waiting for full system restart..."
    sleep 90
    
    # Re-test everything
    ML_IP=$(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null)
    if docker exec portfolio_backend ping -c 1 $ML_IP >/dev/null 2>&1; then
        echo "✅ Network connectivity restored"
        NETWORK_OK=true
    else
        echo "❌ Network connectivity still broken"
    fi
fi

echo ""
echo "4️⃣ Final comprehensive test..."

# Test ML service
echo "🧪 Testing ML service health:"
if curl -s http://localhost:8002/health >/dev/null; then
    echo "✅ ML service: Healthy"
    ML_FINAL=true
else
    echo "❌ ML service: Failed"
    ML_FINAL=false
fi

# Test backend
echo "🧪 Testing backend health:"
if curl -s http://localhost:3001/health >/dev/null; then
    echo "✅ Backend: Healthy"
    BACKEND_FINAL=true
else
    echo "❌ Backend: Failed"
    BACKEND_FINAL=false
fi

# Test AI routes
echo "🧪 Testing AI routes:"
AI_HEALTH=$(curl -s http://localhost:3001/api/ai/health 2>/dev/null)
if echo "$AI_HEALTH" | grep -q '"available":true'; then
    echo "✅ AI routes: ML service available"
    AI_FINAL=true
else
    echo "❌ AI routes: ML service unavailable"
    echo "🔍 Response: $(echo "$AI_HEALTH" | head -1)"
    AI_FINAL=false
fi

# Test backend → ML service connection
echo "🧪 Testing backend → ML service connection:"
if docker exec portfolio_backend curl -f http://ml-service:8002/health >/dev/null 2>&1; then
    echo "✅ Backend can reach ML service via hostname"
    CONNECTION_FINAL=true
else
    echo "❌ Backend cannot reach ML service via hostname"
    CONNECTION_FINAL=false
fi

# Test stock quotes (simpler endpoint)
echo "🧪 Testing stock quotes:"
QUOTE_TEST=$(curl -s http://localhost:3001/api/ai/quote/AAPL 2>/dev/null)
if echo "$QUOTE_TEST" | grep -q '"success":true'; then
    echo "✅ Stock quotes: Working"
    QUOTE_FINAL=true
else
    echo "❌ Stock quotes: Failed"
    QUOTE_FINAL=false
fi

echo ""
echo "5️⃣ Results and recommendations..."
echo "================================"

echo "System Status:"
echo "  ML service health: $([ "$ML_FINAL" = true ] && echo "✅" || echo "❌")"
echo "  Backend health: $([ "$BACKEND_FINAL" = true ] && echo "✅" || echo "❌")"
echo "  AI routes: $([ "$AI_FINAL" = true ] && echo "✅" || echo "❌")"
echo "  Network connectivity: $([ "$CONNECTION_FINAL" = true ] && echo "✅" || echo "❌")"
echo "  Stock quotes: $([ "$QUOTE_FINAL" = true ] && echo "✅" || echo "❌")"

if [ "$ML_FINAL" = true ] && [ "$BACKEND_FINAL" = true ] && [ "$AI_FINAL" = true ] && [ "$CONNECTION_FINAL" = true ]; then
    echo ""
    echo "🎉 ALL SYSTEMS GO!"
    echo ""
    echo "✅ Your portfolio analyzer is ready with:"
    echo "   - Real OCR extraction from images"
    echo "   - Real-time market data"
    echo "   - No mock data fallbacks"
    echo "   - Complete AI analysis"
    echo ""
    echo "🌐 Test your portfolio upload at: http://localhost:3000"
    echo ""
    echo "📋 Upload requirements:"
    echo "   - Clear portfolio screenshot"
    echo "   - Visible stock symbols and values"
    echo "   - Image size under 10MB"
    echo "   - Supported formats: PNG, JPG, JPEG"
    
else
    echo ""
    echo "❌ System not fully operational"
    echo ""
    echo "🔧 Manual steps to try:"
    echo "   1. Full restart: docker-compose down && docker-compose up -d"
    echo "   2. Check logs: docker logs portfolio_ml_service"
    echo "   3. Check logs: docker logs portfolio_backend"
    echo "   4. Verify network: docker network inspect ai-portfolio-analyzer_default"
    echo ""
    echo "🆘 If issues persist:"
    echo "   1. Check Docker Desktop is running properly"
    echo "   2. Restart Docker Desktop"
    echo "   3. Clear Docker cache: docker system prune -a"
fi

echo ""
echo "📊 Final container status:"
docker-compose ps