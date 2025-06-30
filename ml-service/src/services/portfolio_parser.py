import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    ROBINHOOD = "robinhood"
    TD_AMERITRADE = "td_ameritrade"
    ETRADE = "etrade"
    WEBULL = "webull"
    FIDELITY = "fidelity"
    SCHWAB = "schwab"
    GENERIC = "generic"

@dataclass
class Holding:
    symbol: str
    quantity: Optional[float] = None
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    average_cost: Optional[float] = None
    gain_loss: Optional[float] = None
    gain_loss_percent: Optional[float] = None
    confidence: float = 0.0

@dataclass
class PortfolioData:
    total_value: Optional[float] = None
    holdings: List[Holding] = None
    broker: BrokerType = BrokerType.GENERIC
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.holdings is None:
            self.holdings = []

class PortfolioParser:
    def __init__(self):
        # Enhanced stock symbol patterns
        self.symbol_patterns = [
            r'\b([A-Z]{2,5})\s+(?=[\d$])',  # Symbol followed by numbers/prices
            r'^([A-Z]{2,5})$',  # Standalone symbols
            r'\b([A-Z]{1,5})\b',  # General uppercase letters
        ]
        
        # Enhanced price patterns
        self.price_patterns = [
            r'\$([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)',  # $1,234.56
            r'\$([0-9]+\.?[0-9]*)',  # $123.45
            r'([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))',  # 1,234.56
            r'([0-9]+\.[0-9]{2})',  # 123.45
        ]
        
        # Enhanced percentage patterns
        self.percentage_patterns = [
            r'([+-]?[0-9]+\.?[0-9]*)\s*%',
            r'([+-]?[0-9]+\.?[0-9]*)\s*percent',
        ]

    def extract_structured_data(self, text: str) -> List[Dict]:
        """
        Extract structured portfolio data from text that appears to be in table format
        """
        lines = text.split('\n')
        holdings = []
        
        # Look for lines that contain stock symbols and associated data
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip header lines
            if any(header in line.lower() for header in ['symbol', 'qty', 'price', 'value', 'gain', 'positions']):
                continue
            
            # Look for patterns that suggest this is a holding line
            # Pattern: SYMBOL followed by numbers/prices
            symbol_match = re.search(r'\b([A-Z]{2,5})\b', line)
            if symbol_match:
                symbol = symbol_match.group(1)
                
                # Skip common false positives
                if symbol in ['USD', 'CAD', 'EUR', 'GBP', 'THE', 'AND', 'FOR', 'YOU', 'ARE', 'NOT', 'BUT', 'CAN', 'ALL', 'NEW', 'GET', 'MAY', 'USE', 'DAY', 'WAY', 'MAN', 'OLD', 'SEE', 'HIM', 'TWO', 'HER', 'HIS', 'SHE', 'NOW', 'ITS', 'WHO', 'DID', 'YES', 'HAS', 'HAD', 'LET', 'PUT', 'SAY', 'TOO', 'WAS', 'WIN', 'YET']:
                    continue
                
                # Extract all numbers and prices from the line
                prices = re.findall(r'\$?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,4})?)', line)
                percentages = re.findall(r'([+-]?[0-9]+\.?[0-9]*)\s*%', line)
                
                # Clean and convert prices
                cleaned_prices = []
                for price in prices:
                    try:
                        cleaned_price = float(price.replace(',', ''))
                        if 0.001 <= cleaned_price <= 10000000:  # Reasonable range
                            cleaned_prices.append(cleaned_price)
                    except ValueError:
                        continue
                
                # Clean and convert percentages
                cleaned_percentages = []
                for pct in percentages:
                    try:
                        cleaned_pct = float(pct)
                        if -100 <= cleaned_pct <= 10000:  # Reasonable range
                            cleaned_percentages.append(cleaned_pct)
                    except ValueError:
                        continue
                
                # Build holding data
                holding_data = {
                    'symbol': symbol,
                    'prices': cleaned_prices,
                    'percentages': cleaned_percentages,
                    'original_line': line
                }
                
                holdings.append(holding_data)
                logger.info(f"Found holding: {symbol} with {len(cleaned_prices)} prices and {len(cleaned_percentages)} percentages")
        
        return holdings

    def parse_structured_portfolio(self, text: str) -> PortfolioData:
        """
        Parse portfolio data using structured extraction
        """
        logger.info("Parsing portfolio with structured parser")
        
        portfolio = PortfolioData(broker=BrokerType.GENERIC)
        
        # Extract structured data
        structured_holdings = self.extract_structured_data(text)
        
        # Find total value - look for large numbers that might be total
        all_prices = []
        for holding in structured_holdings:
            all_prices.extend(holding['prices'])
        
        if all_prices:
            # Total value is likely the largest number
            potential_totals = [p for p in all_prices if p > 1000]
            if potential_totals:
                portfolio.total_value = max(potential_totals)
                logger.info(f"Identified total portfolio value: ${portfolio.total_value}")
        
        # Convert to Holding objects
        for holding_data in structured_holdings:
            symbol = holding_data['symbol']
            prices = holding_data['prices']
            percentages = holding_data['percentages']
            
            holding = Holding(
                symbol=symbol,
                confidence=0.8  # Higher confidence for structured data
            )
            
            # Assign prices based on typical portfolio table structure
            # Usually: quantity, market_value, current_price, avg_price
            if len(prices) >= 1:
                holding.quantity = prices[0] if prices[0] < 10000 else None  # Quantity usually smaller
            if len(prices) >= 2:
                holding.market_value = prices[1] if prices[1] > prices[0] else None
            if len(prices) >= 3:
                holding.current_price = prices[2]
            if len(prices) >= 4:
                holding.average_cost = prices[3]
            
            # Assign percentages (usually gain/loss percentages)
            if percentages:
                holding.gain_loss_percent = percentages[0]
            
            # Calculate gain/loss if we have enough data
            if holding.market_value and holding.quantity and holding.average_cost:
                cost_basis = holding.quantity * holding.average_cost
                holding.gain_loss = holding.market_value - cost_basis
            
            portfolio.holdings.append(holding)
        
        # Calculate portfolio confidence
        if portfolio.holdings:
            avg_confidence = sum(h.confidence for h in portfolio.holdings) / len(portfolio.holdings)
            portfolio.confidence = avg_confidence * 0.9  # Slightly lower due to parsing complexity
        else:
            portfolio.confidence = 0.3
        
        logger.info(f"Structured parsing complete: {len(portfolio.holdings)} holdings")
        return portfolio

    def parse_generic(self, text: str) -> PortfolioData:
        """
        Fallback generic parser
        """
        logger.info("Using generic parser as fallback")
        
        portfolio = PortfolioData(broker=BrokerType.GENERIC)
        
        # Simple symbol extraction
        symbols = set()
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if self.is_valid_symbol(match):
                    symbols.add(match.upper())
        
        # Create basic holdings
        for symbol in list(symbols)[:15]:  # Limit to 15 holdings
            holding = Holding(
                symbol=symbol,
                confidence=0.5
            )
            portfolio.holdings.append(holding)
        
        portfolio.confidence = 0.4
        return portfolio

    def is_valid_symbol(self, symbol: str) -> bool:
        """
        Validate if a string is likely a stock symbol
        """
        exclude_list = {
            'USD', 'CAD', 'EUR', 'GBP', 'JPY', 'AM', 'PM', 'EST', 'PST',
            'BUY', 'SELL', 'HOLD', 'ETF', 'REIT', 'IRA', 'DIV', 'EPS', 'PE',
            'TODAY', 'WEEK', 'MONTH', 'YEAR', 'THE', 'AND', 'FOR', 'YOU', 
            'ARE', 'NOT', 'BUT', 'CAN', 'ALL', 'NEW', 'GET', 'MAY', 'USE',
            'DAY', 'WAY', 'MAN', 'OLD', 'SEE', 'HIM', 'TWO', 'HER', 'HIS',
            'SHE', 'NOW', 'ITS', 'WHO', 'DID', 'YES', 'HAS', 'HAD', 'LET',
            'PUT', 'SAY', 'TOO', 'WAS', 'WIN', 'YET', 'QTY', 'MKT', 'VAL',
            'AVG', 'PRI', 'LAST', 'OPEN', 'MARK'
        }
        
        if symbol in exclude_list:
            return False
        
        if len(symbol) < 2 or len(symbol) > 5:
            return False
            
        if not symbol.isalpha():
            return False
            
        return True

    async def parse_portfolio(self, ocr_result: Dict) -> PortfolioData:
        """
        Parse portfolio data from OCR result
        """
        try:
            text = ocr_result.get('text', '')
            if not text:
                return PortfolioData(confidence=0.0)
            
            logger.info(f"Parsing text with {len(text)} characters")
            
            # Try structured parsing first
            portfolio = self.parse_structured_portfolio(text)
            
            # If structured parsing didn't find much, try generic
            if len(portfolio.holdings) < 3:
                logger.info("Structured parsing found few holdings, trying generic parser")
                generic_portfolio = self.parse_generic(text)
                if len(generic_portfolio.holdings) > len(portfolio.holdings):
                    portfolio = generic_portfolio
            
            # Adjust confidence based on OCR quality
            ocr_confidence = ocr_result.get('confidence', 0)
            portfolio.confidence *= ocr_confidence
            
            logger.info(f"Final portfolio: {len(portfolio.holdings)} holdings, confidence: {portfolio.confidence:.2f}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Portfolio parsing failed: {e}")
            return PortfolioData(confidence=0.0)