const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();

// Configure multer for memory storage
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  },
});

// Upload and process portfolio screenshot with real-time market data
router.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    console.log('Processing portfolio upload with market data:', {
      filename: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype
    });

    // Create FormData to send to ML service
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    // Call ML service for OCR processing WITH market data
    const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://ml-service:8002';
    
    try {
      console.log('Calling ML service with market data at:', `${mlServiceUrl}/api/ocr/parse-portfolio-with-market-data`);
      
      const mlResponse = await axios.post(
        `${mlServiceUrl}/api/ocr/parse-portfolio-with-market-data`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
          timeout: 90000, // 90 second timeout for OCR + market data
        }
      );

      const { ocr_result, portfolio } = mlResponse.data;

      console.log('ML service response with market data:', {
        text_length: ocr_result.text?.length || 0,
        confidence: ocr_result.confidence,
        holdings_count: portfolio.holdings_count,
        live_total_value: portfolio.live_total_value,
        holdings_with_live_data: portfolio.holdings_with_live_data
      });

      res.json({
        success: true,
        message: 'Portfolio processed successfully with real-time market data',
        data: {
          filename: req.file.originalname,
          size: req.file.size,
          ocr_result: {
            text: ocr_result.text || '',
            confidence: ocr_result.confidence || 0,
            method: ocr_result.method || 'unknown',
            word_count: ocr_result.word_count || 0
          },
          portfolio: {
            // Original parsed data
            total_value: portfolio.total_value || null,
            broker: portfolio.broker || 'unknown',
            holdings_count: portfolio.holdings_count || 0,
            confidence: portfolio.confidence || 0,
            
            // Live market data
            live_total_value: portfolio.live_total_value || null,
            live_total_gain_loss: portfolio.live_total_gain_loss || null,
            live_total_gain_loss_percent: portfolio.live_total_gain_loss_percent || null,
            holdings_with_live_data: portfolio.holdings_with_live_data || 0,
            market_data_timestamp: portfolio.market_data_timestamp,
            
            // Holdings with enhanced data
            holdings: portfolio.holdings || []
          }
        }
      });

    } catch (mlError) {
      console.error('ML service error:', mlError.message);
      
      // Fallback to basic processing without market data
      res.json({
        success: true,
        message: 'Portfolio processed successfully (basic mode - market data unavailable)',
        data: {
          filename: req.file.originalname,
          size: req.file.size,
          ocr_result: {
            text: 'Could not process image with OCR',
            confidence: 0,
            method: 'fallback'
          },
          portfolio: {
            total_value: null,
            broker: 'fallback',
            holdings_count: 0,
            holdings: []
          }
        }
      });
    }

  } catch (error) {
    console.error('Portfolio upload error:', error);
    res.status(500).json({ 
      success: false,
      error: 'Upload processing failed',
      message: error.message 
    });
  }
});

// Get real-time quote for a specific symbol
router.get('/quote/:symbol', async (req, res) => {
  try {
    const symbol = req.params.symbol.toUpperCase();
    const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://ml-service:8002';
    
    const response = await axios.post(`${mlServiceUrl}/api/market/quote/${symbol}`);
    
    res.json(response.data);
    
  } catch (error) {
    console.error('Quote fetch error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch stock quote',
      symbol: req.params.symbol
    });
  }
});

// Health check for portfolio service
router.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    service: 'portfolio-upload-with-market-data',
    timestamp: new Date().toISOString()
  });
});

module.exports = router;