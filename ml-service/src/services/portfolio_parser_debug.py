# Enhanced Portfolio Parser with comprehensive debugging
# Place this in ml-service/src/services/portfolio_parser_debug.py

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DebugPortfolioParser:
    """Enhanced portfolio parser with extensive debugging capabilities"""
    
    def __init__(self):
        # Comprehensive stock symbol patterns for different brokers
        self.stock_patterns = [
            # Pattern 1: SYMBOL SHARES PRICE
            r'([A-Z]{1,5})\s+(\d+(?:,\d{3})*)\s+\$?([\d,]+\.?\d*)',
            
            # Pattern 2: SYMBOL $VALUE
            r'([A-Z]{1,5})\s+\$?([\d,]+\.\d{2})',
            
            # Pattern 3: SYMBOL PERCENTAGE VALUE
            r'([A-Z]{1,5})\s+[+-]?[\d.]+%\s+\$?([\d,]+\.?\d*)',
            
            # Pattern 4: SYMBOL followed by numbers
            r'([A-Z]{2,5})\s+([\d,]+\.?\d*)',
            
            # Pattern 5: More flexible pattern for any stock-like text
            r'\b([A-Z]{2,5})\b.*?\$?([\d,]+(?:\.\d{2})?)',
            
            # Pattern 6: Robinhood style
            r'([A-Z]{1,5})\s+([+-]?\$?[\d,]+\.?\d*)\s+([+-]?[\d.]+%)',
            
            # Pattern 7: Schwab/Fidelity style
            r'([A-Z]{1,5})\s+(\d+)\s+shares?\s+\$?([\d,]+\.?\d*)',
            
            # Pattern 8: E*Trade style with market value
            r'([A-Z]{1,5})\s+.*?Market\s+Value:?\s*\$?([\d,]+\.?\d*)',
        ]
        
        # Common words to exclude from being treated as stock symbols
        self.exclude_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
            'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW',
            'OLD', 'SEE', 'TWO', 'WHO', 'BOY', 'DID', 'SHE', 'USE', 'HER', 'HOW', 'ITS', 'OUR',
            'OUT', 'USE', 'YOU', 'ALL', 'BOY', 'CAN', 'DID', 'GET', 'HAS', 'HAD', 'HER', 'HIM',
            'HIS', 'HOW', 'ITS', 'LET', 'MAY', 'NEW', 'NOT', 'NOW', 'OLD', 'OUT', 'PUT', 'RUN',
            'SAY', 'SHE', 'TOO', 'TWO', 'USE', 'WAS', 'WAY', 'WHO', 'WIN', 'YES', 'YET', 'YOU',
            'QTY', 'SHARES', 'PRICE', 'VALUE', 'TOTAL', 'GAIN', 'LOSS', 'PCT', 'MARKET', 'DAY',
            'LAST', 'OPEN', 'HIGH', 'LOW', 'BID', 'ASK', 'VOL', 'AVG', 'ACCOUNT', 'BALANCE',
            'PORTFOLIO', 'HOLDINGS', 'CASH', 'AVAILABLE', 'BUYING', 'POWER', 'UNREALIZED'
        }
    
    async def debug_parse_portfolio(self, ocr_result: Dict) -> Dict:
        """
        Enhanced parsing with step-by-step debugging
        """
        debug_info = {
            'original_text': ocr_result.get('text', ''),
            'text_length': len(ocr_result.get('text', '')),
            'ocr_confidence': ocr_result.get('confidence', 0),
            'ocr_method': ocr_result.get('method', 'unknown'),
            'preprocessing_steps': [],
            'pattern_matches': {},
            'potential_symbols': [],
            'filtered_symbols': [],
            'final_holdings': [],
            'parsing_issues': []
        }
        
        text = ocr_result.get('text', '')
        
        if not text:
            debug_info['parsing_issues'].append("No text extracted from OCR")
            return self._create_debug_response([], debug_info)
        
        logger.info(f"🔍 DEBUG: Starting to parse text with {len(text)} characters")
        logger.info(f"📝 DEBUG: First 200 chars: {repr(text[:200])}")
        
        # Step 1: Text preprocessing
        debug_info['preprocessing_steps'].append("Starting text preprocessing")
        
        # Clean and normalize text
        cleaned_text = self._preprocess_text(text)
        debug_info['preprocessing_steps'].append(f"Cleaned text length: {len(cleaned_text)}")
        
        # Split into lines for analysis
        lines = cleaned_text.split('\n')
        debug_info['preprocessing_steps'].append(f"Split into {len(lines)} lines")
        
        # Step 2: Try each pattern
        all_potential_holdings = []
        
        for i, pattern in enumerate(self.stock_patterns):
            debug_info['preprocessing_steps'].append(f"Trying pattern {i+1}: {pattern}")
            
            matches = []
            for line_num, line in enumerate(lines):
                line_matches = re.findall(pattern, line, re.IGNORECASE)
                for match in line_matches:
                    matches.append({
                        'match': match,
                        'line_num': line_num,
                        'line_text': line.strip(),
                        'pattern_index': i
                    })
            
            debug_info['pattern_matches'][f'pattern_{i+1}'] = {
                'pattern': pattern,
                'matches_found': len(matches),
                'matches': matches[:5]  # First 5 matches for debugging
            }
            
            # Process matches for this pattern
            for match_info in matches:
                holding = self._process_match(match_info['match'], match_info['line_text'], i)
                if holding:
                    holding['debug_info'] = {
                        'pattern_used': i + 1,
                        'line_number': match_info['line_num'],
                        'original_line': match_info['line_text']
                    }
                    all_potential_holdings.append(holding)
        
        debug_info['potential_symbols'] = [h['symbol'] for h in all_potential_holdings]
        logger.info(f"🔍 DEBUG: Found {len(all_potential_holdings)} potential holdings: {debug_info['potential_symbols']}")
        
        # Step 3: Filter and validate holdings
        debug_info['preprocessing_steps'].append("Filtering and validating holdings")
        
        validated_holdings = []
        for holding in all_potential_holdings:
            if self._validate_symbol(holding['symbol']):
                validated_holdings.append(holding)
                debug_info['filtered_symbols'].append(holding['symbol'])
            else:
                debug_info['parsing_issues'].append(f"Filtered out invalid symbol: {holding['symbol']}")
        
        # Step 4: Remove duplicates and merge data
        final_holdings = self._deduplicate_holdings(validated_holdings)
        debug_info['final_holdings'] = final_holdings
        
        logger.info(f"🎯 DEBUG: Final holdings count: {len(final_holdings)}")
        for holding in final_holdings:
            logger.info(f"   📊 {holding['symbol']}: ${holding.get('market_value', 'N/A')}")
        
        return self._create_debug_response(final_holdings, debug_info)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for better parsing"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Replace common OCR errors
        replacements = {
            '|': 'I',  # Common OCR error
            '0': 'O',  # In stock symbols
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            '"': '',
            '"': '',
            ''': "'",
            ''': "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _process_match(self, match: tuple, line_text: str, pattern_index: int) -> Optional[Dict]:
        """Process a regex match into a holding dictionary"""
        try:
            if len(match) < 2:
                return None
            
            symbol = match[0].upper().strip()
            
            # Extract numerical values from the match
            values = []
            for item in match[1:]:
                try:
                    # Clean and convert to float
                    clean_value = re.sub(r'[^\d.]', '', str(item))
                    if clean_value and '.' in clean_value or clean_value.isdigit():
                        values.append(float(clean_value))
                except:
                    continue
            
            if not values:
                return None
            
            # Create holding object with best guess at values
            holding = {
                'symbol': symbol,
                'market_value': max(values) if values else 0,  # Assume largest value is market value
                'shares': min(values) if len(values) > 1 else 0,  # Assume smallest is shares
                'price': values[0] if len(values) == 1 else (max(values) / min(values) if min(values) > 0 else 0),
                'raw_match': match,
                'extraction_confidence': 0.7  # Default confidence
            }
            
            return holding
            
        except Exception as e:
            logger.error(f"Error processing match {match}: {e}")
            return None
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate if a string looks like a stock symbol"""
        if not symbol or len(symbol) < 1:
            return False
        
        # Must be 1-5 characters
        if len(symbol) < 1 or len(symbol) > 5:
            return False
        
        # Must be alphabetic
        if not symbol.isalpha():
            return False
        
        # Must be uppercase
        if symbol != symbol.upper():
            return False
        
        # Must not be in exclude list
        if symbol in self.exclude_words:
            return False
        
        return True
    
    def _deduplicate_holdings(self, holdings: List[Dict]) -> List[Dict]:
        """Remove duplicate holdings and merge data"""
        symbol_map = {}
        
        for holding in holdings:
            symbol = holding['symbol']
            if symbol not in symbol_map:
                symbol_map[symbol] = holding
            else:
                # Merge data - keep the one with higher market value
                existing = symbol_map[symbol]
                if holding.get('market_value', 0) > existing.get('market_value', 0):
                    symbol_map[symbol] = holding
        
        return list(symbol_map.values())
    
    def _create_debug_response(self, holdings: List[Dict], debug_info: Dict) -> Dict:
        """Create a comprehensive debug response"""
        
        # Calculate portfolio totals
        total_value = sum(h.get('market_value', 0) for h in holdings)
        
        return {
            'success': True,
            'holdings': holdings,
            'holdings_count': len(holdings),
            'total_value': total_value,
            'broker': 'generic',
            'extraction_method': 'enhanced_regex_with_debug',
            'confidence': 0.8 if holdings else 0.1,
            'debug_info': debug_info,
            'troubleshooting': self._generate_troubleshooting_tips(debug_info),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_troubleshooting_tips(self, debug_info: Dict) -> List[str]:
        """Generate specific troubleshooting tips based on parsing results"""
        tips = []
        
        if debug_info['text_length'] == 0:
            tips.append("❌ No text was extracted - check OCR service and image quality")
        elif debug_info['text_length'] < 50:
            tips.append("⚠️ Very little text extracted - image may be too small or low quality")
        
        if debug_info['ocr_confidence'] < 0.5:
            tips.append("⚠️ Low OCR confidence - try a clearer, higher resolution image")
        
        if not debug_info['potential_symbols']:
            tips.append("❌ No stock symbols detected - ensure image shows portfolio with stock tickers")
            tips.append("💡 Look for symbols like AAPL, MSFT, TSLA, etc. in the image")
        
        if debug_info['potential_symbols'] and not debug_info['filtered_symbols']:
            tips.append("❌ All symbols were filtered out - check for common words being detected as symbols")
        
        total_matches = sum(data['matches_found'] for data in debug_info['pattern_matches'].values())
        if total_matches == 0:
            tips.append("❌ No pattern matches found - the portfolio format may not be supported")
            tips.append("💡 Supported formats: Robinhood, Schwab, Fidelity, E*Trade, TD Ameritrade")
        
        return tips

# Updated main parsing function for integration
async def debug_parse_portfolio_data(ocr_result: Dict) -> Dict:
    """
    Main parsing function with debugging - use this to replace the existing parser
    """
    parser = DebugPortfolioParser()
    result = await parser.debug_parse_portfolio(ocr_result)
    
    # Log extensive debugging information
    logger.info("=" * 80)
    logger.info("🔍 PORTFOLIO PARSING DEBUG REPORT")
    logger.info("=" * 80)
    logger.info(f"📝 Original text length: {result['debug_info']['text_length']}")
    logger.info(f"🎯 OCR confidence: {result['debug_info']['ocr_confidence']:.2f}")
    logger.info(f"🔢 Potential symbols found: {len(result['debug_info']['potential_symbols'])}")
    logger.info(f"✅ Valid symbols after filtering: {len(result['debug_info']['filtered_symbols'])}")
    logger.info(f"📊 Final holdings: {result['holdings_count']}")
    
    if result['debug_info']['parsing_issues']:
        logger.warning("⚠️ Parsing issues:")
        for issue in result['debug_info']['parsing_issues']:
            logger.warning(f"   - {issue}")
    
    if result['debug_info']['potential_symbols']:
        logger.info(f"🔍 Potential symbols: {result['debug_info']['potential_symbols']}")
    
    if result['debug_info']['filtered_symbols']:
        logger.info(f"✅ Filtered symbols: {result['debug_info']['filtered_symbols']}")
    
    # Log first 500 characters of original text for debugging
    original_text = result['debug_info']['original_text']
    if original_text:
        logger.info(f"📄 First 500 chars of OCR text:")
        logger.info(f"   {repr(original_text[:500])}")
    
    logger.info("=" * 80)
    
    return result