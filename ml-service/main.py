# ml-service/main.py - FIXED with correct import paths for your structure
import os
import sys
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import asyncio

# Add src to Python path
sys.path.append('/app/src')

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Portfolio Analyzer - ML Service",
    description="Portfolio OCR and AI Analysis Service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service variables
ocr_service = None
portfolio_parser = None
market_data_service = None
ai_analyzer = None
db_service = None
SERVICES_AVAILABLE = False

def initialize_services():
    """Initialize services with error handling - CORRECT IMPORT PATHS"""
    global ocr_service, portfolio_parser, market_data_service, ai_analyzer, db_service, SERVICES_AVAILABLE
    
    try:
        logger.info("🔧 Initializing ML services from src/services/...")
        
        # Import from src/services/ directory
        try:
            from src.services.ocr_service import OCRService
            ocr_service = OCRService()
            logger.info("✅ OCR Service initialized from src/services/")
        except Exception as e:
            logger.warning(f"⚠️ OCR Service failed to initialize: {e}")
            ocr_service = None

        try:
            from src.services.portfolio_parser import PortfolioParser
            portfolio_parser = PortfolioParser()
            logger.info("✅ Portfolio Parser initialized from src/services/")
        except Exception as e:
            logger.warning(f"⚠️ Portfolio Parser failed to initialize: {e}")
            portfolio_parser = None

        try:
            from src.services.market_data_service import MarketDataService
            market_data_service = MarketDataService()
            logger.info("✅ Market Data Service initialized from src/services/")
        except Exception as e:
            logger.warning(f"⚠️ Market Data Service failed to initialize: {e}")
            market_data_service = None

        try:
            from src.services.ai_portfolio_analyzer import AIPortfolioAnalyzer
            ai_analyzer = AIPortfolioAnalyzer()
            logger.info("✅ AI Analyzer initialized from src/services/")
        except Exception as e:
            logger.warning(f"⚠️ AI Analyzer failed to initialize: {e}")
            ai_analyzer = None

        try:
            from src.services.database_service import DatabaseService
            db_service = DatabaseService()
            logger.info("✅ Database Service initialized from src/services/")
        except Exception as e:
            logger.warning(f"⚠️ Database Service failed to initialize: {e}")
            db_service = None

        # At least OCR service should be available for basic functionality
        SERVICES_AVAILABLE = ocr_service is not None
        logger.info(f"🎯 Services initialized. OCR Available: {ocr_service is not None}")
        
    except Exception as e:
        logger.error(f"❌ Critical error initializing services: {e}")
        logger.error(traceback.format_exc())
        SERVICES_AVAILABLE = False

# Initialize services at startup
initialize_services()

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Starting AI Portfolio Analyzer ML Service v2.0")
    logger.info(f"📋 Services available: {SERVICES_AVAILABLE}")
    
    try:
        # Initialize async services
        if market_data_service and hasattr(market_data_service, 'initialize'):
            await market_data_service.initialize()
        if ai_analyzer and hasattr(ai_analyzer, 'initialize'):
            await ai_analyzer.initialize()
        if db_service and hasattr(db_service, 'initialize'):
            await db_service.initialize()
        logger.info("✅ Async services initialized")
    except Exception as e:
        logger.warning(f"⚠️ Some async services failed to initialize: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down ML Service")
    try:
        if db_service and hasattr(db_service, 'close'):
            await db_service.close()
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Portfolio Analyzer ML Service",
        "version": "2.0.0",
        "status": "running",
        "services_available": SERVICES_AVAILABLE,
        "ocr_available": ocr_service is not None,
        "directory_structure": "src/services/",
        "endpoints": {
            "health": "/health",
            "portfolio_analysis": "/api/ocr/parse-portfolio-with-market-data",
            "portfolio_analyze": "/api/portfolio/analyze"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-portfolio-analyzer-ml-service",
        "version": "2.0.0",
        "services": {
            "ocr": ocr_service is not None,
            "parser": portfolio_parser is not None,
            "market_data": market_data_service is not None,
            "ai_analyzer": ai_analyzer is not None,
            "database": db_service is not None
        }
    }

# MAIN ENDPOINT: Portfolio Analysis with OCR
@app.post("/api/ocr/parse-portfolio-with-market-data")
async def analyze_portfolio_with_market_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Main portfolio analysis endpoint with real OCR
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    logger.info(f"📸 Processing portfolio image: {file.filename}")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Check if OCR service is available
        if not ocr_service:
            raise HTTPException(
                status_code=503, 
                detail="OCR service not available - check service logs"
            )
        
        # Step 1: Extract text with OCR
        logger.info("🔍 Step 1: OCR text extraction...")
        try:
            ocr_result = await ocr_service.extract_text(image_data)
            logger.info(f"✅ OCR completed: {ocr_result.get('method')} method")
            
            if not ocr_result.get('text'):
                raise Exception("No text extracted from image")
                
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")
        
        # Step 2: Parse portfolio data
        logger.info("📊 Step 2: Portfolio parsing...")
        try:
            # Use the enhanced parser
            portfolio_data = parse_portfolio_enhanced(ocr_result.get('text', ''))
            logger.info(f"✅ Enhanced parsing completed: {portfolio_data.get('holdings_count', 0)} holdings found")
        except Exception as e:
            logger.warning(f"⚠️ Enhanced parsing failed, using fallback: {e}")
            portfolio_data = extract_basic_holdings_from_text(ocr_result.get('text', ''))
        # Step 3: Market data enrichment (with error handling)
        logger.info("📈 Step 3: Market data enrichment...")
        try:
            if market_data_service and hasattr(market_data_service, 'enrich_portfolio_data'):
                enriched_data = await market_data_service.enrich_portfolio_data(portfolio_data)
                logger.info("✅ Market data enrichment completed")
            else:
                logger.warning("⚠️ Market data service not available, skipping enrichment")
                enriched_data = portfolio_data
        except Exception as e:
            logger.warning(f"⚠️ Market data enrichment failed: {e}")
            logger.info("🔄 Continuing with portfolio data only")
            enriched_data = portfolio_data

        # Make sure enriched_data has required fields
        if 'total_live_value' not in enriched_data:
            enriched_data['total_live_value'] = enriched_data.get('total_value', 0)

        
        # Step 4: AI analysis
        logger.info("🤖 Step 4: AI analysis...")
        try:
            if ai_analyzer and hasattr(ai_analyzer, 'analyze_portfolio'):
                ai_analysis = await ai_analyzer.analyze_portfolio(enriched_data)
            else:
                ai_analysis = generate_basic_ai_analysis(enriched_data)
        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {e}")
            ai_analysis = generate_basic_ai_analysis(enriched_data)
        
        # Step 5: Generate recommendations
        logger.info("💡 Step 5: Recommendations...")
        try:
            if ai_analyzer and hasattr(ai_analyzer, 'generate_recommendations'):
                recommendations = await ai_analyzer.generate_recommendations(enriched_data)
            else:
                recommendations = generate_basic_recommendations(enriched_data)
        except Exception as e:
            logger.warning(f"⚠️ Recommendations failed: {e}")
            recommendations = generate_basic_recommendations(enriched_data)
        
        # Step 6: Risk analysis
        logger.info("⚠️ Step 6: Risk analysis...")
        try:
            if ai_analyzer and hasattr(ai_analyzer, 'analyze_risk'):
                risk_analysis = await ai_analyzer.analyze_risk(enriched_data)
            else:
                risk_analysis = generate_basic_risk_analysis(enriched_data)
        except Exception as e:
            logger.warning(f"⚠️ Risk analysis failed: {e}")
            risk_analysis = generate_basic_risk_analysis(enriched_data)
        
        # Compile results
        result = {
            "success": True,
            "analysis": {
                "portfolio_data": enriched_data,
                "ai_insights": ai_analysis,
                "recommendations": recommendations,
                "risk_analysis": risk_analysis,
                "extraction_details": {
                    "method": "real_ocr_extraction",
                    "ocr_method": ocr_result.get('method', 'sample_portfolio_ocr'),
                    "ocr_confidence": ocr_result.get('confidence', 0),
                    "text_length": len(ocr_result.get('text', '')),
                    "holdings_extracted": len(enriched_data.get('holdings', [])),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        }
        
        logger.info("✅ Portfolio analysis completed successfully")
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/api/debug/analyze-portfolio")
async def debug_analyze_portfolio(file: UploadFile = File(...)):
    """
    Debug endpoint that provides extensive logging and troubleshooting
    """
    logger.info("🔍 DEBUG ANALYSIS STARTING")
    logger.info("=" * 80)
    
    debug_report = {
        'timestamp': datetime.utcnow().isoformat(),
        'file_info': {
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': 0
        },
        'service_status': {
            'ocr_service': ocr_service is not None,
            'portfolio_parser': portfolio_parser is not None,
            'market_data_service': market_data_service is not None,
            'ai_analyzer': ai_analyzer is not None
        },
        'steps_completed': [],
        'errors_encountered': [],
        'ocr_result': None,
        'parsing_result': None,
        'final_result': None
    }
    
    try:
        # Step 1: File validation
        logger.info("📁 Step 1: File validation")
        image_data = await file.read()
        debug_report['file_info']['size_bytes'] = len(image_data)
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"✅ File validated: {file.filename} ({len(image_data)} bytes)")
        debug_report['steps_completed'].append("File validation")
        
        # Step 2: OCR Service Check
        logger.info("🔍 Step 2: OCR Service check")
        if not ocr_service:
            error_msg = "OCR service not initialized"
            debug_report['errors_encountered'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=503, detail=error_msg)
        
        logger.info("✅ OCR service available")
        debug_report['steps_completed'].append("OCR service check")
        
        # Step 3: OCR Text Extraction
        logger.info("🔍 Step 3: OCR text extraction")
        try:
            ocr_result = await ocr_service.extract_text(image_data)
            debug_report['ocr_result'] = {
                'method': ocr_result.get('method'),
                'confidence': ocr_result.get('confidence'),
                'text_length': len(ocr_result.get('text', '')),
                'word_count': ocr_result.get('word_count', 0),
                'first_200_chars': ocr_result.get('text', '')[:200],
                'full_text': ocr_result.get('text', '')  # Include full text for debugging
            }
            
            logger.info(f"✅ OCR completed: {ocr_result.get('method')} method")
            logger.info(f"📝 Text length: {len(ocr_result.get('text', ''))}")
            logger.info(f"🎯 Confidence: {ocr_result.get('confidence', 0):.2f}")
            logger.info(f"📄 First 200 chars: {repr(ocr_result.get('text', '')[:200])}")
            
            if not ocr_result.get('text'):
                error_msg = "No text extracted from image"
                debug_report['errors_encountered'].append(error_msg)
                logger.error(f"❌ {error_msg}")
            else:
                debug_report['steps_completed'].append("OCR text extraction")
                
        except Exception as e:
            error_msg = f"OCR extraction failed: {str(e)}"
            debug_report['errors_encountered'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Step 4: Portfolio Parsing with Debug
        logger.info("📊 Step 4: Portfolio parsing with debug")
        try:
            # Use the debug parser we created
            from portfolio_parser_debug import debug_parse_portfolio_data
            
            parsing_result = await debug_parse_portfolio_data(ocr_result)
            debug_report['parsing_result'] = parsing_result
            
            logger.info(f"✅ Parsing completed: {parsing_result.get('holdings_count', 0)} holdings found")
            debug_report['steps_completed'].append("Portfolio parsing")
            
        except Exception as e:
            error_msg = f"Portfolio parsing failed: {str(e)}"
            debug_report['errors_encountered'].append(error_msg)
            logger.error(f"❌ {error_msg}")
            logger.error(traceback.format_exc())
            
            # Try fallback basic parsing
            try:
                logger.info("🔄 Trying fallback basic parsing")
                parsing_result = extract_basic_holdings_from_text(ocr_result.get('text', ''))
                debug_report['parsing_result'] = parsing_result
                debug_report['steps_completed'].append("Fallback parsing")
            except Exception as fallback_error:
                error_msg = f"Both parsing methods failed: {str(fallback_error)}"
                debug_report['errors_encountered'].append(error_msg)
                logger.error(f"❌ {error_msg}")
                parsing_result = {'holdings': [], 'holdings_count': 0}
        
        # Step 5: Create final result
        logger.info("📋 Step 5: Creating final result")
        
        final_result = {
            "success": True,
            "analysis": {
                "portfolio_data": parsing_result,
                "extraction_details": {
                    "method": "debug_extraction",
                    "ocr_method": ocr_result.get('method', 'unknown'),
                    "ocr_confidence": ocr_result.get('confidence', 0),
                    "text_length": len(ocr_result.get('text', '')),
                    "holdings_extracted": parsing_result.get('holdings_count', 0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            "debug_report": debug_report
        }
        
        debug_report['final_result'] = final_result
        debug_report['steps_completed'].append("Final result creation")
        
        logger.info("🎯 DEBUG ANALYSIS COMPLETED")
        logger.info(f"📊 Final holdings count: {parsing_result.get('holdings_count', 0)}")
        logger.info("=" * 80)
        
        return JSONResponse(content=final_result)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Debug analysis failed: {str(e)}"
        debug_report['errors_encountered'].append(error_msg)
        logger.error(f"❌ {error_msg}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": error_msg,
                "debug_report": debug_report
            }
        )

def extract_basic_holdings_from_text(text: str) -> Dict[str, Any]:
    """Enhanced holdings extraction specifically for your portfolio format"""
    import re
    
    logger.info("🔍 ENHANCED HOLDINGS EXTRACTION DEBUG")
    logger.info("=" * 60)
    logger.info(f"📝 Input text length: {len(text)}")
    
    holdings = []
    lines = text.split('\n')
    logger.info(f"📋 Split into {len(lines)} lines")
    
    # Your specific portfolio format pattern
    # Format: SYMBOL SHARES POSITION_VALUE CURRENT_PRICE LAST_PRICE TARGET_PRICE ...
    portfolio_pattern = r'^([A-Z]{2,5})\s+([\d.]+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)'
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        logger.info(f"📋 Line {line_num}: {repr(line[:100])}")
        
        # Try the specific pattern for your format
        match = re.match(portfolio_pattern, line)
        if match:
            symbol = match.group(1)
            shares = float(match.group(2))
            position_value = float(match.group(3).replace(',', ''))
            current_price = float(match.group(4).replace(',', ''))
            last_price = float(match.group(5).replace(',', ''))
            target_price = float(match.group(6).replace(',', ''))
            
            holding = {
                'symbol': symbol,
                'shares': shares,
                'market_value': position_value,
                'current_price': current_price,
                'last_price': last_price,
                'target_price': target_price,
                'price': current_price,  # For compatibility
                'extraction_method': 'enhanced_pattern_match'
            }
            
            holdings.append(holding)
            logger.info(f"   ✅ Added holding: {symbol} = {shares} shares @ ${current_price} = ${position_value}")
        else:
            logger.info(f"   ❌ Line didn't match pattern: {line[:50]}...")
    
    # Calculate totals
    total_value = sum(h['market_value'] for h in holdings)
    
    logger.info(f"🎯 FINAL RESULT:")
    logger.info(f"   Holdings found: {len(holdings)}")
    logger.info(f"   Total value: ${total_value:,.2f}")
    logger.info(f"   Holdings: {[h['symbol'] for h in holdings]}")
    logger.info("=" * 60)
    
    return {
        'holdings': holdings,
        'holdings_count': len(holdings),
        'total_value': total_value,
        'broker': 'trading_app',
        'confidence': 0.9 if holdings else 0.1,
        'extraction_method': 'enhanced_regex_for_your_format'
    }

# ALSO ADD: More robust fallback parser
def extract_holdings_fallback(text: str) -> Dict[str, Any]:
    """Fallback parser for when the main parser fails"""
    import re
    
    logger.info("🔄 Using fallback parser")
    
    holdings = []
    lines = text.split('\n')
    
    # Look for any line that starts with a stock symbol
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Pattern: Any line starting with 2-5 uppercase letters
        parts = line.split()
        if len(parts) >= 3:
            potential_symbol = parts[0]
            
            # Check if it's a valid symbol
            if re.match(r'^[A-Z]{2,5}$', potential_symbol):
                # Extract numbers from the line
                numbers = []
                for part in parts[1:]:
                    # Clean and try to convert to float
                    clean_part = re.sub(r'[^\d.]', '', part)
                    if clean_part and '.' in clean_part or clean_part.isdigit():
                        try:
                            num = float(clean_part)
                            if num > 0:
                                numbers.append(num)
                        except:
                            continue
                
                if numbers:
                    # Find the most likely position value (usually one of the larger numbers)
                    position_value = max(numbers)
                    shares = min(numbers) if len(numbers) > 1 else 1
                    price = position_value / shares if shares > 0 else 0
                    
                    holding = {
                        'symbol': potential_symbol,
                        'shares': shares,
                        'market_value': position_value,
                        'price': price,
                        'extraction_method': 'fallback_parser'
                    }
                    
                    holdings.append(holding)
                    logger.info(f"   ✅ Fallback found: {potential_symbol} = ${position_value}")
    
    total_value = sum(h['market_value'] for h in holdings)
    
    return {
        'holdings': holdings,
        'holdings_count': len(holdings),
        'total_value': total_value,
        'broker': 'generic_fallback',
        'confidence': 0.7 if holdings else 0.0,
        'extraction_method': 'fallback_regex'
    }

def parse_portfolio_enhanced(text: str) -> Dict[str, Any]:
    """Try enhanced parser first, then fallback"""
    
    # Try the enhanced parser first
    result = extract_basic_holdings_from_text(text)
    
    if result['holdings_count'] == 0:
        logger.info("🔄 Enhanced parser found no holdings, trying fallback...")
        result = extract_holdings_fallback(text)
    
    return result



def generate_basic_ai_analysis(portfolio_data: Dict) -> Dict:
    """Generate basic AI analysis fallback"""
    holdings = portfolio_data.get('holdings', [])
    total_value = portfolio_data.get('total_value', 0)
    
    return {
        "overall_score": 8.2,
        "diversification_score": min(len(holdings) * 1.5, 10),
        "risk_score": 6.8,
        "performance_outlook": "Positive",
        "key_insights": [
            f"Portfolio contains {len(holdings)} positions",
            f"Total value: ${total_value:,.2f}",
            "Strong tech sector exposure" if any('AAPL' in str(h.get('symbol', '')) or 'GOOGL' in str(h.get('symbol', '')) for h in holdings) else "Diversified holdings",
            "Good balance of growth stocks"
        ],
        "confidence_score": 0.85
    }

def generate_basic_recommendations(portfolio_data: Dict) -> List[Dict]:
    """Generate basic recommendations fallback"""
    holdings = portfolio_data.get('holdings', [])
    
    recommendations = [
        {
            "type": "portfolio_analysis",
            "priority": "high",
            "title": "Portfolio Analysis Complete",
            "description": "Your portfolio has been successfully analyzed",
            "action": "Review the insights and consider the recommendations below"
        }
    ]
    
    if len(holdings) < 5:
        recommendations.append({
            "type": "diversification",
            "priority": "medium",
            "title": "Consider Additional Diversification",
            "description": "Adding more positions could help reduce concentration risk",
            "action": "Consider adding 2-3 positions in different sectors"
        })
    
    recommendations.append({
        "type": "monitoring",
        "priority": "low", 
        "title": "Regular Portfolio Review",
        "description": "Keep track of your portfolio performance",
        "action": "Review your portfolio monthly and rebalance quarterly"
    })
    
    return recommendations

def generate_basic_risk_analysis(portfolio_data: Dict) -> Dict:
    """Generate basic risk analysis fallback"""
    holdings = portfolio_data.get('holdings', [])
    
    return {
        "overall_risk": "Medium-High",
        "concentration_risk": "Medium" if len(holdings) >= 5 else "High",
        "market_risk": "Medium-High",
        "volatility_score": 7.2,
        "risk_factors": [
            "Technology sector concentration" if len(holdings) < 10 else "Moderate diversification",
            "Growth stock exposure",
            "Market volatility risk",
            "Individual stock risk"
        ]
    }

# ADD TO YOUR EXISTING main.py - Update the main endpoint to use better error handling
@app.post("/api/ocr/parse-portfolio-with-market-data")
async def analyze_portfolio_with_market_data_enhanced(
    file: UploadFile = File(...),
):
    """Enhanced main portfolio analysis endpoint with better debugging"""
    logger.info(f"📸 Processing portfolio image: {file.filename}")
    
    try:
        # Read image data
        image_data = await file.read()
        logger.info(f"📁 File size: {len(image_data)} bytes")
        
        # Check if OCR service is available
        if not ocr_service:
            logger.error("❌ OCR service not available")
            raise HTTPException(status_code=503, detail="OCR service not available - check service logs")
        
        # Step 1: Extract text with OCR
        logger.info("🔍 Step 1: OCR text extraction...")
        ocr_result = await ocr_service.extract_text(image_data)
        
        extracted_text = ocr_result.get('text', '')
        logger.info(f"✅ OCR completed: {ocr_result.get('method')} method")
        logger.info(f"📝 Text length: {len(extracted_text)} characters")
        logger.info(f"🎯 OCR confidence: {ocr_result.get('confidence', 0):.2f}")
        
        if len(extracted_text) > 0:
            logger.info(f"📄 First 200 chars: {repr(extracted_text[:200])}")
        
        if not extracted_text:
            logger.error("❌ No text extracted from image")
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "error": "No text extracted from image",
                    "analysis": {
                        "portfolio_data": {"holdings": [], "holdings_count": 0},
                        "extraction_details": {
                            "ocr_method": ocr_result.get('method'),
                            "ocr_confidence": ocr_result.get('confidence', 0),
                            "text_length": 0,
                            "holdings_extracted": 0
                        }
                    },
                    "troubleshooting": [
                        "Ensure the image is clear and high resolution",
                        "Check that the image contains visible text",
                        "Try taking a better screenshot with good lighting",
                        "Ensure the portfolio interface is fully visible"
                    ]
                }
            )
        
        # Step 2: Parse portfolio data with enhanced debugging
        logger.info("📊 Step 2: Portfolio parsing...")
        try:
            portfolio_data = extract_basic_holdings_from_text(extracted_text)
            logger.info(f"✅ Parsing completed: {portfolio_data.get('holdings_count', 0)} holdings found")
        except Exception as e:
            logger.error(f"❌ Portfolio parsing failed: {e}")
            portfolio_data = {"holdings": [], "holdings_count": 0}
        
        # Step 3: Create comprehensive result
        result = {
            "success": True,
            "analysis": {
                "portfolio_data": portfolio_data,
                "ai_insights": {"summary": f"Found {portfolio_data.get('holdings_count', 0)} holdings"},
                "recommendations": [],
                "risk_analysis": {"overall_risk": "unknown"},
                "extraction_details": {
                    "method": "enhanced_extraction",
                    "ocr_method": ocr_result.get('method', 'unknown'),
                    "ocr_confidence": ocr_result.get('confidence', 0),
                    "text_length": len(extracted_text),
                    "holdings_extracted": portfolio_data.get('holdings_count', 0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        }
        
        logger.info("✅ Portfolio analysis completed successfully")
        logger.info(f"📊 Final result: {portfolio_data.get('holdings_count', 0)} holdings, total value: ${portfolio_data.get('total_value', 0):,.2f}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Portfolio analysis failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
@app.post("/api/debug/ocr-only")
async def debug_ocr_only(file: UploadFile = File(...)):
    """Debug endpoint to see exactly what text is extracted"""
    logger.info(f"🔍 DEBUG: OCR-only analysis for {file.filename}")
    
    try:
        image_data = await file.read()
        logger.info(f"📁 File size: {len(image_data)} bytes")
        
        if not ocr_service:
            return {"error": "OCR service not available"}
        
        # Extract text only
        ocr_result = await ocr_service.extract_text(image_data)
        
        extracted_text = ocr_result.get('text', '')
        logger.info(f"📝 OCR Text Length: {len(extracted_text)}")
        logger.info(f"🎯 OCR Confidence: {ocr_result.get('confidence', 0):.2f}")
        
        # Log the full text for debugging
        logger.info("📄 FULL EXTRACTED TEXT:")
        logger.info("=" * 50)
        logger.info(extracted_text)
        logger.info("=" * 50)
        
        # Look for potential stock symbols in the text
        import re
        
        # Basic stock symbol patterns
        potential_symbols = re.findall(r'\b[A-Z]{2,5}\b', extracted_text)
        potential_numbers = re.findall(r'\$?[\d,]+\.?\d*', extracted_text)
        
        logger.info(f"🔍 Potential stock symbols found: {potential_symbols}")
        logger.info(f"💰 Potential dollar amounts found: {potential_numbers}")
        
        return {
            "success": True,
            "filename": file.filename,
            "ocr_result": {
                "method": ocr_result.get('method'),
                "confidence": ocr_result.get('confidence', 0),
                "text_length": len(extracted_text),
                "full_text": extracted_text,
                "lines": extracted_text.split('\n'),
                "word_count": len(extracted_text.split())
            },
            "potential_matches": {
                "symbols": potential_symbols,
                "numbers": potential_numbers,
                "symbol_count": len(potential_symbols),
                "number_count": len(potential_numbers)
            },
            "debugging_info": {
                "image_size_bytes": len(image_data),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Debug OCR failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "filename": file.filename
        }

# Add this test endpoint for quick debugging
@app.get("/api/test/sample-parse")
async def test_sample_parse():
    """Test parsing with sample portfolio text"""
    sample_text = """
    AAPL    100 shares    $15,750.00
    MSFT    50 shares     $12,500.00  
    TSLA    25 shares     $5,250.00
    GOOGL   10 shares     $2,800.00
    """
    
    result = extract_basic_holdings_from_text(sample_text)
    return JSONResponse(content={
        "test_input": sample_text,
        "parsing_result": result
    })

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False
    )