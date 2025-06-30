from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

# Fix the imports - use relative imports
from services.ocr_service import OCRService
from services.portfolio_parser import PortfolioParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Portfolio Analyzer - ML Service",
    description="Machine Learning service for portfolio screenshot analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ocr_service = OCRService()
portfolio_parser = PortfolioParser()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ml-service"
    }

@app.get("/")
async def root():
    return {"message": "AI Portfolio Analyzer ML Service"}

@app.post("/api/ocr/extract")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using OCR
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Extract text using OCR
        ocr_result = await ocr_service.extract_text(image_data)
        
        return JSONResponse(content={
            "success": True,
            "result": ocr_result,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/api/ocr/debug")
async def debug_ocr(file: UploadFile = File(...)):
    """
    Debug OCR extraction - returns detailed results
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        # Get both extraction methods
        layout_result = await ocr_service.extract_text_with_layout(image_data)
        standard_result = await ocr_service.extract_text_tesseract(image_data)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "layout_extraction": {
                "text": layout_result.get('text', ''),
                "confidence": layout_result.get('confidence', 0),
                "method": layout_result.get('method', 'unknown'),
                "lines": layout_result.get('lines', [])
            },
            "standard_extraction": {
                "text": standard_result.get('text', ''),
                "confidence": standard_result.get('confidence', 0),
                "method": standard_result.get('method', 'unknown')
            }
        })
        
    except Exception as e:
        logger.error(f"Debug OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debug OCR failed: {str(e)}")

@app.post("/api/ocr/parse-portfolio")
async def parse_portfolio_from_image(file: UploadFile = File(...)):
    """
    Extract and parse portfolio data from image
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        logger.info(f"Processing image: {file.filename}, size: {len(image_data)} bytes")
        
        # Extract text using OCR
        ocr_result = await ocr_service.extract_text(image_data)
        
        logger.info(f"OCR extracted {len(ocr_result.get('text', ''))} characters with {ocr_result.get('confidence', 0):.2f} confidence")
        
        # Parse portfolio data
        portfolio_data = await portfolio_parser.parse_portfolio(ocr_result)
        
        logger.info(f"Parsed portfolio: {len(portfolio_data.holdings)} holdings")
        
        return JSONResponse(content={
            "success": True,
            "ocr_result": {
                "text": ocr_result.get('text', ''),
                "confidence": ocr_result.get('confidence', 0),
                "method": ocr_result.get('method', 'unknown'),
                "word_count": ocr_result.get('word_count', 0)
            },
            "portfolio": {
                "total_value": portfolio_data.total_value,
                "broker": portfolio_data.broker.value,
                "holdings_count": len(portfolio_data.holdings),
                "holdings": [
                    {
                        "symbol": holding.symbol,
                        "quantity": holding.quantity,
                        "current_price": holding.current_price,
                        "market_value": holding.market_value,
                        "gain_loss": holding.gain_loss,
                        "gain_loss_percent": holding.gain_loss_percent,
                        "confidence": holding.confidence
                    }
                    for holding in portfolio_data.holdings
                ],
                "confidence": portfolio_data.confidence
            },
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"Portfolio parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio parsing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)