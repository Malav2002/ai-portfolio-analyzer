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

// Upload and process portfolio screenshot
router.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    console.log('Processing portfolio upload:', {
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

    // Call ML service for OCR processing
    const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://ml-service:8002';
    
    try {
      console.log('Calling ML service at:', `${mlServiceUrl}/api/ocr/parse-portfolio`);
      
      const mlResponse = await axios.post(
        `${mlServiceUrl}/api/ocr/parse-portfolio`,
        formData,
        {
          headers: {
            ...formData.getHeaders(),
          },
          timeout: 60000, // 60 second timeout for OCR processing
        }
      );

      const { ocr_result, portfolio } = mlResponse.data;

      console.log('ML service response:', {
        text_length: ocr_result.text?.length || 0,
        confidence: ocr_result.confidence,
        holdings_count: portfolio.holdings_count
      });

      res.json({
        success: true,
        message: 'Portfolio processed successfully with real OCR',
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
            total_value: portfolio.total_value || null,
            broker: portfolio.broker || 'unknown',
            holdings_count: portfolio.holdings_count || 0,
            holdings: portfolio.holdings || [],
            confidence: portfolio.confidence || 0
          }
        }
      });

    } catch (mlError) {
      console.error('ML service error:', mlError.message);
      
      // Fallback to demo data if ML service fails
      res.json({
        success: true,
        message: 'Portfolio processed successfully (fallback mode)',
        data: {
          filename: req.file.originalname,
          size: req.file.size,
          ocr_result: {
            text: 'FALLBACK: Could not process image with OCR',
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

// Health check for portfolio service
router.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy',
    service: 'portfolio-upload',
    timestamp: new Date().toISOString()
  });
});

module.exports = router;