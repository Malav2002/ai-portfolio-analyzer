#!/bin/bash


echo "🎯 Targeted ML Service Connection Fix"
echo "===================================="

echo "1️⃣ Current status check..."
echo "🔍 Testing AI routes directly:"
AI_RESPONSE=$(curl -s http://localhost:3001/api/ai/health)
echo "$AI_RESPONSE"

echo ""
echo "🔍 Checking if ML service is really unavailable:"
if echo "$AI_RESPONSE" | grep -q '"ml_service_available":false'; then
    echo "❌ Backend reports ML service as unavailable"
    
    echo ""
    echo "2️⃣ Testing ML service directly..."
    if curl -s http://localhost:8002/health > /dev/null; then
        echo "✅ ML service responds directly"
        
        echo ""
        echo "3️⃣ The issue is in backend's ML service check logic"
        echo "Let's check backend logs for ML service connection attempts:"
        docker logs portfolio_backend --tail 20 | grep -i "ml\|service\|8002" || echo "No ML service connection logs found"
        
        echo ""
        echo "4️⃣ Testing the exact URL backend should use..."
        echo "🔗 Testing http://ml-service:8002 from host (simulated):"
        
        # Get ML service IP and test
        ML_IP=$(docker inspect portfolio_ml_service --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
        echo "📍 ML service IP: $ML_IP"
        
        if curl -s http://$ML_IP:8002/health > /dev/null; then
            echo "✅ ML service reachable via IP"
        else
            echo "❌ ML service not reachable via IP"
        fi
        
        echo ""
        echo "5️⃣ Checking backend's ML_SERVICE_URL configuration..."
        ML_URL=$(docker exec portfolio_backend env | grep ML_SERVICE_URL || echo "ML_SERVICE_URL not set")
        echo "🔧 Backend ML_SERVICE_URL: $ML_URL"
        
        if echo "$ML_URL" | grep -q "ml-service:8002"; then
            echo "✅ ML_SERVICE_URL is correctly configured"
        else
            echo "❌ ML_SERVICE_URL is incorrect or missing"
            echo "🔧 Expected: ML_SERVICE_URL=http://ml-service:8002"
        fi
        
        echo ""
        echo "6️⃣ Testing backend's ability to resolve ml-service hostname..."
        # Test if backend can resolve the hostname
        if docker exec portfolio_backend nslookup ml-service 2>/dev/null | grep -q "Address"; then
            echo "✅ Backend can resolve ml-service hostname"
        else
            echo "❌ Backend cannot resolve ml-service hostname"
            echo "🔧 This indicates a Docker networking issue"
        fi
        
        echo ""
        echo "7️⃣ Force backend to check ML service again..."
        echo "🔄 Restarting backend to refresh connections..."
        docker-compose restart backend
        
        echo "⏱️ Waiting for backend to restart..."
        sleep 15
        
        echo "🧪 Testing AI routes after restart..."
        NEW_AI_RESPONSE=$(curl -s http://localhost:3001/api/ai/health)
        
        if echo "$NEW_AI_RESPONSE" | grep -q '"ml_service_available":true'; then
            echo "🎉 SUCCESS! ML service now available"
            echo "✅ Backend can now reach ML service"
        else
            echo "❌ Still showing ML service unavailable"
            echo "🔍 Backend may need ML service URL fix"
            
            echo ""
            echo "8️⃣ Applying ML service URL fix..."
            
            # Check if .env has correct ML_SERVICE_URL
            if [ -f ".env" ]; then
                if grep -q "ML_SERVICE_URL=http://ml-service:8002" .env; then
                    echo "✅ .env has correct ML_SERVICE_URL"
                else
                    echo "🔧 Fixing ML_SERVICE_URL in .env"
                    sed -i.bak 's|ML_SERVICE_URL=.*|ML_SERVICE_URL=http://ml-service:8002|' .env
                    echo "ML_SERVICE_URL=http://ml-service:8002" >> .env
                fi
            else
                echo "📝 Creating .env with ML_SERVICE_URL"
                echo "ML_SERVICE_URL=http://ml-service:8002" > .env
            fi
            
            echo "🔄 Restarting backend with corrected environment..."
            docker-compose stop backend
            docker-compose up -d backend
            
            echo "⏱️ Waiting for backend restart..."
            sleep 20
            
            echo "🧪 Final test..."
            FINAL_AI_RESPONSE=$(curl -s http://localhost:3001/api/ai/health)
            
            if echo "$FINAL_AI_RESPONSE" | grep -q '"ml_service_available":true'; then
                echo "🎉 SUCCESS! ML service connection fixed"
            else
                echo "❌ Still having issues - manual intervention needed"
                echo "🔍 Response: $FINAL_AI_RESPONSE"
            fi
        fi
        
    else
        echo "❌ ML service not responding directly"
        echo "🔄 Restarting ML service..."
        docker-compose restart ml-service
        
        echo "⏱️ Waiting for ML service restart..."
        sleep 30
        
        if curl -s http://localhost:8002/health > /dev/null; then
            echo "✅ ML service now responding"
            docker-compose restart backend
            sleep 15
        else
            echo "❌ ML service still not responding"
            echo "📋 ML service logs:"
            docker logs portfolio_ml_service --tail 20
        fi
    fi
    
elif echo "$AI_RESPONSE" | grep -q '"ml_service_available":true'; then
    echo "🎉 ML service is already available!"
    echo "✅ No fix needed - try uploading a portfolio now"
else
    echo "⚠️ Unexpected AI response format"
    echo "Response: $AI_RESPONSE"
fi

echo ""
echo "9️⃣ Final verification..."
echo "🧪 Testing complete chain:"

# Test ML service
if curl -s http://localhost:8002/health > /dev/null; then
    echo "✅ ML service: OK"
else
    echo "❌ ML service: Failed"
fi

# Test backend
if curl -s http://localhost:3001/health > /dev/null; then
    echo "✅ Backend: OK"
else
    echo "❌ Backend: Failed"
fi

# Test AI routes
AI_FINAL=$(curl -s http://localhost:3001/api/ai/health)
if echo "$AI_FINAL" | grep -q '"ml_service_available":true'; then
    echo "✅ AI routes: ML service available"
    echo ""
    echo "🎉 PORTFOLIO ANALYZER READY!"
    echo "🌐 Upload a portfolio image at: http://localhost:3000"
elif echo "$AI_FINAL" | grep -q '"ml_service_available":false'; then
    echo "❌ AI routes: ML service unavailable"
    echo "🔍 Need manual debugging"
else
    echo "⚠️ AI routes: Unexpected response"
fi

echo ""
echo "📊 Container status:"
docker-compose ps