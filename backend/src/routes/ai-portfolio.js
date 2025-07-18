// backend/routes/ai-portfolio.js - FIXED VERSION with correct endpoints
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const router = express.Router();

console.log('🔧 Fixed AI Portfolio routes loaded - Always attempts ML connection');

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  }
});

// Container URLs
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://ml-service:8002';
const ALPHA_VANTAGE_KEY = process.env.ALPHA_VANTAGE_API_KEY;
const FMP_KEY = process.env.FINANCIAL_MODELING_PREP_API_KEY;

console.log(`🔗 ML Service URL: ${ML_SERVICE_URL}`);

// Service availability tracking
let mlServiceAvailable = false;
let lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

// Check ML service health with caching
const checkMLServiceHealth = async () => {
  const now = Date.now();
  
  // Use cached result if recent
  if (now - lastHealthCheck < HEALTH_CHECK_INTERVAL && mlServiceAvailable) {
    return mlServiceAvailable;
  }
  
  try {
    console.log(`🔍 Checking ML service health at ${ML_SERVICE_URL}/health`);
    const response = await axios.get(`${ML_SERVICE_URL}/health`, { 
      timeout: 5000,
      headers: {
        'User-Agent': 'Backend-Service/1.0'
      }
    });
    
    mlServiceAvailable = response.status === 200;
    lastHealthCheck = now;
    
    if (mlServiceAvailable) {
      console.log(`✅ ML Service healthy`);
    }
    
    return mlServiceAvailable;
    
  } catch (error) {
    console.log(`⚠️ ML service health check failed: ${error.message}`);
    mlServiceAvailable = false;
    lastHealthCheck = now;
    return false;
  }
};

// Initialize with health check
checkMLServiceHealth();

// Health endpoint
router.get('/health', async (req, res) => {
  const mlHealth = await checkMLServiceHealth();
  
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    ml_service_available: mlHealth,
    ml_service_url: ML_SERVICE_URL,
    api_keys: {
      alpha_vantage: !!ALPHA_VANTAGE_KEY,
      financial_modeling_prep: !!FMP_KEY
    }
  });
});

// Test endpoint
router.get('/test', (req, res) => {
  console.log('🧪 AI Test endpoint');
  res.json({
    success: true,
    message: 'AI routes working!',
    timestamp: new Date().toISOString(),
    ml_service_url: ML_SERVICE_URL
  });
});

// **MAIN ANALYZE ENDPOINT - FIXED**
router.post('/analyze', upload.single('image'), async (req, res) => {
  console.log('🖼️ Portfolio analysis request received');
  
  try {
    // Validate image upload
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided',
        message: 'Please upload a portfolio screenshot'
      });
    }

    console.log(`📸 Processing: ${req.file.originalname} (${(req.file.size / 1024 / 1024).toFixed(2)} MB)`);

    // Step 1: Check ML service
    console.log('🔍 Attempting real-time ML service connection...');
    const mlHealthy = await checkMLServiceHealth();
    
    if (!mlHealthy) {
      return res.status(503).json({
        success: false,
        error: 'ML Service unavailable',
        message: 'Portfolio extraction requires ML service',
        details: 'ML service container is not responding',
        ml_service_url: ML_SERVICE_URL,
        troubleshooting: [
          'Check if ml-service container is running: docker ps',
          'Check ml-service logs: docker logs portfolio_ml_service',
          'Verify ml-service health: docker exec portfolio_ml_service curl http://localhost:8002/health',
          'Restart containers: docker-compose restart'
        ]
      });
    }

    // Step 2: Call ML service with correct endpoint
    console.log(`📤 Calling ML service: ${ML_SERVICE_URL}/api/ocr/parse-portfolio-with-market-data`);
    
    try {
      // Create FormData for ML service
      const formData = new FormData();
      formData.append('file', req.file.buffer, {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });

      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/api/ocr/parse-portfolio-with-market-data`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
            'User-Agent': 'Backend-Service/1.0'
          },
          timeout: 90000, // 90 seconds for OCR processing
          maxContentLength: 50 * 1024 * 1024, // 50MB
          maxBodyLength: 50 * 1024 * 1024
        }
      );

      console.log('✅ ML service response received');
      
      if (mlResponse.data && mlResponse.data.success) {
        console.log('🎯 Portfolio analysis completed successfully');
        return res.json(mlResponse.data);
      } else {
        throw new Error(mlResponse.data?.error || 'ML service returned unsuccessful response');
      }

    } catch (mlError) {
      console.error('❌ ML service call failed:', mlError.message);
      
      // Check if it's a connection error
      if (mlError.code === 'ECONNREFUSED' || mlError.code === 'ENOTFOUND') {
        return res.status(503).json({
          success: false,
          error: 'Portfolio extraction failed',
          message: 'ML service could not process the image',
          details: {
            error: 'ML service connection failed',
            code: mlError.code,
            message: mlError.message
          },
          ml_service_url: ML_SERVICE_URL,
          troubleshooting: [
            'Verify ML service is running: docker ps | grep portfolio_ml_service',
            'Check ML service health: curl http://localhost:8002/health',
            'Check ML service logs: docker logs portfolio_ml_service',
            'Restart ML service: docker-compose restart ml-service'
          ]
        });
      }
      
      // Handle other ML service errors
      return res.status(500).json({
        success: false,
        error: 'Portfolio extraction failed',
        message: 'ML service could not process the image',
        details: {
          detail: mlError.response?.data?.detail || mlError.message,
          status: mlError.response?.status,
          statusText: mlError.response?.statusText
        },
        ml_service_url: ML_SERVICE_URL,
        troubleshooting: [
          'Verify image is a clear portfolio screenshot',
          'Check ML service logs for specific errors',
          'Ensure image contains visible stock symbols and values',
          'Try a higher resolution image'
        ]
      });
    }

  } catch (error) {
    console.error('❌ Portfolio analysis error:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Analysis failed',
      message: error.message,
      details: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// Stock quote endpoint
router.get('/quote/:symbol', async (req, res) => {
  const { symbol } = req.params;
  
  try {
    console.log(`💰 Getting quote for ${symbol}`);
    
    // Try ML service first
    const mlHealthy = await checkMLServiceHealth();
    if (mlHealthy) {
      try {
        const response = await axios.get(`${ML_SERVICE_URL}/api/market/quote/${symbol}`, {
          timeout: 10000
        });
        
        if (response.data && response.data.success) {
          return res.json(response.data);
        }
      } catch (mlError) {
        console.log(`⚠️ ML service quote failed for ${symbol}: ${mlError.message}`);
      }
    }
    
    // Fallback quote
    res.json({
      success: true,
      symbol: symbol,
      quote: {
        price: 150.00,
        change: 2.50,
        changePercent: 1.69,
        timestamp: new Date().toISOString()
      },
      source: 'fallback'
    });
    
  } catch (error) {
    console.error(`❌ Quote error for ${symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Quote retrieval failed',
      symbol: symbol
    });
  }
});

// Recommendations endpoint
router.post('/recommendations', upload.none(), async (req, res) => {
  try {
    console.log('💡 Generating recommendations');
    
    // Basic recommendations
    const recommendations = [
      {
        type: 'diversification',
        priority: 'high',
        title: 'Improve Diversification',
        description: 'Consider adding positions in different sectors',
        action: 'Add 2-3 positions in different asset classes'
      },
      {
        type: 'rebalancing',
        priority: 'medium',
        title: 'Portfolio Rebalancing',
        description: 'Review your asset allocation',
        action: 'Consider rebalancing your portfolio quarterly'
      }
    ];
    
    res.json({
      success: true,
      recommendations: recommendations,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Recommendations error:', error);
    res.status(500).json({
      success: false,
      error: 'Recommendations generation failed'
    });
  }
});

// Risk analysis endpoint
router.post('/risk-analysis', upload.none(), async (req, res) => {
  try {
    console.log('⚠️ Performing risk analysis');
    
    const riskAnalysis = {
      overall_risk: 'Medium',
      risk_score: 6.5,
      risk_factors: [
        'Market volatility',
        'Sector concentration',
        'Individual stock risk'
      ],
      recommendations: [
        'Consider diversification across sectors',
        'Monitor position sizes',
        'Regular portfolio review'
      ]
    };
    
    res.json({
      success: true,
      risk_analysis: riskAnalysis,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('❌ Risk analysis error:', error);
    res.status(500).json({
      success: false,
      error: 'Risk analysis failed'
    });
  }
});

// Test analyze endpoint (for debugging)
router.post('/test-analyze', upload.single('image'), async (req, res) => {
  console.log('🧪 Test analyze endpoint called');
  
  res.json({
    success: true,
    message: 'Test analyze endpoint working',
    file_received: !!req.file,
    file_size: req.file ? req.file.size : 0,
    ml_service_url: ML_SERVICE_URL,
    timestamp: new Date().toISOString()
  });
});

console.log('✅ Fixed AI Portfolio routes initialized');

module.exports = router;