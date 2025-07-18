// backend/routes/portfolio.js - Missing routes file
const express = require('express');
const multer = require('multer');
const router = express.Router();

console.log('📁 Portfolio routes loaded');

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

// Health check
router.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'portfolio-routes',
    timestamp: new Date().toISOString()
  });
});

// Upload endpoint (basic version - the main functionality is in ai-portfolio.js)
router.post('/upload', upload.single('image'), async (req, res) => {
  console.log('📸 Portfolio upload endpoint called');
  
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided'
      });
    }

    console.log(`📸 File received: ${req.file.originalname} (${req.file.size} bytes)`);

    // This is a basic endpoint - main analysis happens in /api/ai/analyze
    res.json({
      success: true,
      message: 'File uploaded successfully',
      filename: req.file.originalname,
      size: req.file.size,
      note: 'Use /api/ai/analyze for full portfolio analysis'
    });

  } catch (error) {
    console.error('❌ Portfolio upload error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get portfolio data endpoint
router.get('/data/:id?', (req, res) => {
  const { id } = req.params;
  
  console.log(`📊 Portfolio data requested for ID: ${id || 'default'}`);
  
  // Sample portfolio data
  const portfolioData = {
    id: id || 'sample',
    holdings: [
      { symbol: 'AAPL', quantity: 15, market_value: 2625.00 },
      { symbol: 'GOOGL', quantity: 8, market_value: 1084.80 },
      { symbol: 'MSFT', quantity: 20, market_value: 6850.00 }
    ],
    total_value: 10559.80,
    last_updated: new Date().toISOString()
  };

  res.json({
    success: true,
    portfolio: portfolioData
  });
});

// List portfolios endpoint
router.get('/list', (req, res) => {
  console.log('📋 Portfolio list requested');
  
  const portfolios = [
    {
      id: 'portfolio_1',
      name: 'Main Portfolio',
      total_value: 10559.80,
      last_updated: new Date().toISOString()
    }
  ];

  res.json({
    success: true,
    portfolios: portfolios
  });
});

console.log('✅ Portfolio routes initialized');

module.exports = router;