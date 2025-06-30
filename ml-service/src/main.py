from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import List

# Import our services
from services.ocr_service import OCRService
from services.portfolio_parser import PortfolioParser
from services.market_data_service import MarketDataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Portfolio Analyzer - ML Service",
    description="Machine Learning service for portfolio screenshot analysis with real-time market data",
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
market_data_service = MarketDataService()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ml-service-with-market-data"
    }

@app.get("/")
async def root():
    return {"message": "AI Portfolio Analyzer ML Service with Real-Time Market Data"}

@app.get("/api/market/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """
    Get real-time quote for a single stock symbol
    """
    try:
        symbol = symbol.upper()
        quote = await market_data_service.get_stock_quote(symbol)
        
        if quote:
            return JSONResponse(content={
                "success": True,
                "symbol": symbol,
                "quote": quote
            })
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Could not fetch quote for {symbol}"
                }
            )
            
    except Exception as e:
        logger.error(f"Quote fetch error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote: {str(e)}")

@app.get("/api/market/quotes")
async def get_multiple_quotes(symbols: List[str]):
    """
    Get real-time quotes for multiple symbols
    """
    try:
        # Clean and validate symbols
        clean_symbols = [s.upper().strip() for s in symbols if s.strip()]
        
        if not clean_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        quotes = await market_data_service.get_multiple_quotes(clean_symbols)
        
        return JSONResponse(content={
            "success": True,
            "symbols_requested": len(clean_symbols),
            "quotes_received": len([q for q in quotes.values() if q.get('current_price')]),
            "quotes": quotes,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Multiple quotes error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch quotes: {str(e)}")

@app.post("/api/ocr/parse-portfolio-with-market-data")
async def parse_portfolio_with_live_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Parse portfolio and enhance with real-time market data
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        logger.info(f"Processing image with market data: {file.filename}")
        
        # Extract text using OCR
        ocr_result = await ocr_service.extract_text(image_data)
        
        # Parse portfolio data
        portfolio_data = await portfolio_parser.parse_portfolio(ocr_result)
        
        # Convert portfolio data to dict for enhancement
        portfolio_dict = {
            "total_value": portfolio_data.total_value,
            "broker": portfolio_data.broker.value,
            "holdings_count": len(portfolio_data.holdings),
            "holdings": [
                {
                    "symbol": holding.symbol,
                    "quantity": holding.quantity,
                    "current_price": holding.current_price,
                    "market_value": holding.market_value,
                    "average_cost": holding.average_cost,
                    "gain_loss": holding.gain_loss,
                    "gain_loss_percent": holding.gain_loss_percent,
                    "confidence": holding.confidence
                }
                for holding in portfolio_data.holdings
            ],
            "confidence": portfolio_data.confidence
        }
        
        # Get symbols for market data
        symbols = [holding.symbol for holding in portfolio_data.holdings if holding.symbol]
        
        if symbols:
            logger.info(f"Fetching live market data for {len(symbols)} symbols")
            market_quotes = await market_data_service.get_multiple_quotes(symbols)
            enhanced_portfolio = market_data_service.enhance_portfolio_with_market_data(
                portfolio_dict, market_quotes
            )
        else:
            enhanced_portfolio = portfolio_dict
        
        return JSONResponse(content={
            "success": True,
            "ocr_result": {
                "text": ocr_result.get('text', ''),
                "confidence": ocr_result.get('confidence', 0),
                "method": ocr_result.get('method', 'unknown'),
                "word_count": ocr_result.get('word_count', 0)
            },
            "portfolio": enhanced_portfolio,
            "filename": file.filename,
            "processing_timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Portfolio parsing with market data failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Keep existing endpoints
@app.post("/api/ocr/extract")
async def extract_text_from_image(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using OCR
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        ocr_result = await ocr_service.extract_text(image_data)
        
        return JSONResponse(content={
            "success": True,
            "result": ocr_result,
            "filename": file.filename
        })
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/api/ocr/parse-portfolio")
async def parse_portfolio_from_image(file: UploadFile = File(...)):
    """
    Extract and parse portfolio data from image (without market data)
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        logger.info(f"Processing image: {file.filename}")
        
        ocr_result = await ocr_service.extract_text(image_data)
        portfolio_data = await portfolio_parser.parse_portfolio(ocr_result)
        
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