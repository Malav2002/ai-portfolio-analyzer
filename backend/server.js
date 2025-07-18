// Fixed server.js with proper route mounting and debugging
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

console.log('🚀 Starting AI Portfolio Analyzer Backend...');

// CORS setup - Allow all origins for development
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000', '*'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Handle preflight requests
app.options('*', cors());

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
app.use((req, res, next) => {
  console.log(`📍 ${new Date().toISOString()} ${req.method} ${req.url}`);
  console.log(`📦 Headers:`, req.headers);
  if (req.body && Object.keys(req.body).length > 0) {
    console.log(`📄 Body:`, Object.keys(req.body));
  }
  next();
});

// Import and mount routes
let portfolioRoutes, aiRoutes;

try {
  portfolioRoutes = require('./routes/portfolio');
  console.log('✅ Portfolio routes loaded successfully');
} catch (error) {
  console.error('❌ Error loading portfolio routes:', error.message);
}

try {
  aiRoutes = require('./routes/ai-portfolio');
  console.log('✅ AI portfolio routes loaded successfully');
} catch (error) {
  console.error('❌ Error loading AI portfolio routes:', error.message);
}

// Mount routes with detailed logging
if (portfolioRoutes) {
  app.use('/api/portfolio', portfolioRoutes);
  console.log('🔗 Portfolio routes mounted at /api/portfolio');
}

if (aiRoutes) {
  app.use('/api/ai', aiRoutes);
  console.log('🔗 AI routes mounted at /api/ai');
  
  // Test the specific analyze route
  console.log('🧪 Testing AI analyze route availability...');
  
  // Add a direct test route for debugging
  app.post('/api/ai/test-analyze', (req, res) => {
    console.log('🧪 Test analyze endpoint called');
    res.json({
      success: true,
      message: 'Test analyze endpoint is working',
      timestamp: new Date().toISOString()
    });
  });
  
} else {
  console.log('❌ AI routes not available - analyze endpoint will not work');
}

// Health check endpoint
app.get('/health', (req, res) => {
  console.log('❤️ Health check requested');
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'ai-portfolio-analyzer-backend',
    version: '2.0.0',
    routes_available: {
      portfolio: !!portfolioRoutes,
      ai: !!aiRoutes
    }
  });
});

// Test endpoint
app.get('/test', (req, res) => {
  console.log('🧪 Test endpoint called');
  res.json({ 
    message: 'Backend is working!', 
    timestamp: new Date().toISOString(),
    ai_routes_available: !!aiRoutes,
    portfolio_routes_available: !!portfolioRoutes
  });
});

// Debug routes endpoint - shows all available routes
app.get('/debug/routes', (req, res) => {
  console.log('🔍 Route debugging requested');
  
  const routes = [];
  
  // Extract routes from the app
  function extractRoutes(app, prefix = '') {
    if (app.router && app.router.stack) {
      app.router.stack.forEach(layer => {
        if (layer.route) {
          // Direct route
          routes.push({
            path: prefix + layer.route.path,
            methods: Object.keys(layer.route.methods).map(m => m.toUpperCase()),
            type: 'direct'
          });
        } else if (layer.name === 'router') {
          // Router middleware
          const routerPrefix = layer.regexp.toString().match(/\^\\?(.*?)\\\?\?\$/);
          const newPrefix = routerPrefix ? routerPrefix[1].replace(/\\\//g, '/') : '';
          
          if (layer.handle && layer.handle.stack) {
            layer.handle.stack.forEach(routeLayer => {
              if (routeLayer.route) {
                routes.push({
                  path: newPrefix + routeLayer.route.path,
                  methods: Object.keys(routeLayer.route.methods).map(m => m.toUpperCase()),
                  type: 'router'
                });
              }
            });
          }
        }
      });
    }
  }
  
  extractRoutes(app);
  
  res.json({ 
    available_routes: routes,
    total_routes: routes.length,
    analyze_endpoint_available: routes.some(route => 
      route.path.includes('/api/ai/analyze') && route.methods.includes('POST')
    )
  });
});

// Specific endpoint test for the analyze route
app.get('/debug/ai-analyze', (req, res) => {
  console.log('🔍 Checking AI analyze endpoint...');
  
  // Check if the route exists
  let analyzeRouteFound = false;
  
  if (app._router && app._router.stack) {
    app._router.stack.forEach(layer => {
      if (layer.name === 'router' && layer.handle && layer.handle.stack) {
        layer.handle.stack.forEach(routeLayer => {
          if (routeLayer.route && 
              routeLayer.route.path === '/analyze' && 
              routeLayer.route.methods.post) {
            analyzeRouteFound = true;
          }
        });
      }
    });
  }
  
  res.json({
    ai_routes_loaded: !!aiRoutes,
    analyze_route_found: analyzeRouteFound,
    expected_endpoint: 'POST /api/ai/analyze',
    suggestion: analyzeRouteFound ? 
      'Route should be working - check frontend URL' : 
      'Route not found - check ai-portfolio.js file'
  });
});

// Catch-all error handler for debugging
app.use('*', (req, res, next) => {
  console.log(`❓ Unmatched route: ${req.method} ${req.originalUrl}`);
  console.log(`📍 Available AI routes should include: POST /api/ai/analyze`);
  res.status(404).json({
    error: 'Route not found',
    requested: `${req.method} ${req.originalUrl}`,
    available_endpoints: [
      'GET /health',
      'GET /test', 
      'GET /debug/routes',
      'GET /debug/ai-analyze',
      'POST /api/ai/analyze (if ai-portfolio.js is working)',
      'POST /api/portfolio/upload (if portfolio.js is working)'
    ]
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('❌ Global error handler:', err);
  res.status(500).json({ 
    success: false,
    error: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});

// Start server with comprehensive logging
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 AI Portfolio Analyzer Backend running on http://0.0.0.0:${PORT}`);
  console.log(`\n📊 Available endpoints:`);
  console.log(`   ❤️  Health: http://localhost:${PORT}/health`);
  console.log(`   🧪 Test: http://localhost:${PORT}/test`);
  console.log(`   🔍 Debug Routes: http://localhost:${PORT}/debug/routes`);
  console.log(`   🔍 Debug AI Analyze: http://localhost:${PORT}/debug/ai-analyze`);
  
  if (portfolioRoutes) {
    console.log(`   📁 Portfolio Upload: POST http://localhost:${PORT}/api/portfolio/upload`);
  }
  
  if (aiRoutes) {
    console.log(`   🤖 AI Endpoints:`);
    console.log(`      ❤️  AI Health: GET http://localhost:${PORT}/api/ai/health`);
    console.log(`      📊 Portfolio Analysis: POST http://localhost:${PORT}/api/ai/analyze`);
    console.log(`      💰 Stock Quote: GET http://localhost:${PORT}/api/ai/quote/AAPL`);
    console.log(`      💡 Recommendations: POST http://localhost:${PORT}/api/ai/recommendations`);
    console.log(`      ⚠️  Risk Analysis: POST http://localhost:${PORT}/api/ai/risk-analysis`);
    console.log(`      🧪 Test Analyze: POST http://localhost:${PORT}/api/ai/test-analyze`);
  } else {
    console.log(`   ❌ AI endpoints not available - check routes/ai-portfolio.js`);
  }
  
  console.log(`\n🔧 Debugging tips:`);
  console.log(`   - Check route availability: curl http://localhost:${PORT}/debug/routes`);
  console.log(`   - Test AI analyze: curl http://localhost:${PORT}/debug/ai-analyze`);
  console.log(`   - Frontend should call: POST http://localhost:${PORT}/api/ai/analyze`);
  
  console.log(`\n✅ Server ready for connections!`);
});

module.exports = app;