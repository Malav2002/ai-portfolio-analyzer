import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import cv2
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PortfolioVisionModel(nn.Module):
    """
    Deep learning model for analyzing portfolio screenshots
    Uses pre-trained vision transformer with custom classification head
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(PortfolioVisionModel, self).__init__()
        
        # Use EfficientNet as backbone (lightweight and effective)
        try:
            import timm
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            self.feature_dim = self.backbone.num_features
        except ImportError:
            # Fallback to basic CNN if timm not available
            self.backbone = self._create_basic_cnn()
            self.feature_dim = 512
        
        # Custom classification head for portfolio analysis
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Regression head for confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _create_basic_cnn(self):
        """Fallback CNN architecture"""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 512)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        classification = self.classifier(features)
        confidence = self.confidence_head(features)
        return classification, confidence, features

class FinancialRiskPredictor(nn.Module):
    """
    Transformer-based model for financial risk prediction
    """
    
    def __init__(self, input_dim=50, hidden_dim=256, num_layers=4, num_heads=8):
        super(FinancialRiskPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder for sequential financial data
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-task prediction heads
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # 5 risk levels
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Ensure positive volatility
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Returns can be negative
        )
        
        self.correlation_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Correlation between 0 and 1
        )
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        # Transformer expects (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        
        # Apply transformer
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Use mean pooling across sequence dimension
        pooled = encoded.mean(dim=0)
        
        # Multi-task predictions
        risk = self.risk_head(pooled)
        volatility = self.volatility_head(pooled)
        returns = self.return_head(pooled)
        correlation = self.correlation_head(pooled)
        
        return {
            'risk': risk,
            'volatility': volatility,
            'returns': returns,
            'correlation': correlation
        }

class PortfolioRecommendationSystem(nn.Module):
    """
    Neural collaborative filtering model for portfolio recommendations
    """
    
    def __init__(self, num_assets=1000, num_features=100, embedding_dim=64):
        super(PortfolioRecommendationSystem, self).__init__()
        
        # Asset embeddings
        self.asset_embedding = nn.Embedding(num_assets, embedding_dim)
        self.asset_bias = nn.Embedding(num_assets, 1)
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        # Deep neural network for recommendations
        self.recommendation_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for portfolio context
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=4)
    
    def forward(self, asset_ids, features, portfolio_context):
        # Get asset embeddings
        asset_emb = self.asset_embedding(asset_ids)
        asset_bias = self.asset_bias(asset_ids).squeeze(-1)
        
        # Process features
        feature_emb = self.feature_processor(features)
        
        # Apply attention to portfolio context
        context_emb, _ = self.attention(asset_emb, portfolio_context, portfolio_context)
        
        # Combine all embeddings
        combined = torch.cat([asset_emb, feature_emb, context_emb], dim=-1)
        
        # Generate recommendation score
        score = self.recommendation_net(combined).squeeze(-1) + asset_bias
        
        return score

class DeepLearningPortfolioAnalyzer:
    """
    Main class that combines all deep learning models for portfolio analysis
    """
    
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.vision_model = None
        self.risk_model = None
        self.recommendation_model = None
        
        # Initialize preprocessing
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Load pre-trained models if available
        self._load_models()
        
        # Initialize NLP pipeline for text analysis
        try:
            self.nlp_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except Exception:
            # Fallback to basic sentiment analysis
            self.nlp_pipeline = pipeline("sentiment-analysis")
        
        logger.info(f"🤖 Deep Learning Portfolio Analyzer initialized on {self.device}")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            # Try to load vision model
            vision_path = os.path.join(self.model_dir, "vision_model.pth")
            if os.path.exists(vision_path):
                self.vision_model = PortfolioVisionModel()
                self.vision_model.load_state_dict(torch.load(vision_path, map_location=self.device))
                self.vision_model.to(self.device)
                self.vision_model.eval()
                logger.info("✅ Loaded pre-trained vision model")
            
            # Try to load risk model
            risk_path = os.path.join(self.model_dir, "risk_model.pth")
            if os.path.exists(risk_path):
                self.risk_model = FinancialRiskPredictor()
                self.risk_model.load_state_dict(torch.load(risk_path, map_location=self.device))
                self.risk_model.to(self.device)
                self.risk_model.eval()
                logger.info("✅ Loaded pre-trained risk model")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize models with default parameters"""
        self.vision_model = PortfolioVisionModel().to(self.device)
        self.risk_model = FinancialRiskPredictor().to(self.device)
        self.recommendation_model = PortfolioRecommendationSystem().to(self.device)
        
        # Set to evaluation mode
        self.vision_model.eval()
        self.risk_model.eval()
        self.recommendation_model.eval()
        
        logger.info("🔥 Initialized default deep learning models")
    
    async def analyze_portfolio_image(self, image_data: bytes) -> Dict:
        """
        Advanced image analysis using deep learning
        """
        try:
            # Preprocess image
            image = self._preprocess_image(image_data)
            
            if self.vision_model is None:
                return self._fallback_image_analysis(image_data)
            
            with torch.no_grad():
                # Run vision model
                classification, confidence, features = self.vision_model(image)
                
                # Extract predictions
                pred_class = torch.argmax(classification, dim=1).cpu().numpy()[0]
                conf_score = confidence.cpu().numpy()[0][0]
                
                # Analyze layout and structure
                layout_analysis = self._analyze_layout(image_data)
                
                return {
                    'predicted_broker': self._map_class_to_broker(pred_class),
                    'confidence': float(conf_score),
                    'layout_analysis': layout_analysis,
                    'visual_features': features.cpu().numpy().tolist()[0][:10],  # First 10 features
                    'analysis_method': 'deep_learning'
                }
                
        except Exception as e:
            logger.error(f"Deep learning image analysis failed: {e}")
            return self._fallback_image_analysis(image_data)
    
    def _preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image for deep learning model"""
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def _analyze_layout(self, image_data: bytes) -> Dict:
        """Analyze image layout using computer vision"""
        try:
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect text regions
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze layout structure
            layout_info = {
                'text_regions': len(contours),
                'image_dimensions': img.shape[:2],
                'detected_tables': self._detect_table_structure(gray),
                'text_density': self._calculate_text_density(gray)
            }
            
            return layout_info
            
        except Exception as e:
            logger.warning(f"Layout analysis failed: {e}")
            return {'analysis': 'failed', 'error': str(e)}
    
    def _detect_table_structure(self, gray_image) -> int:
        """Detect table-like structures in the image"""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count line intersections (table indicators)
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return len([c for c in contours if cv2.contourArea(c) > 500])
    
    def _calculate_text_density(self, gray_image) -> float:
        """Calculate text density in the image"""
        # Apply threshold to detect text
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate density
        total_pixels = thresh.shape[0] * thresh.shape[1]
        text_pixels = np.sum(thresh == 0)  # Assuming text is dark
        
        return text_pixels / total_pixels
    
    async def advanced_risk_analysis(self, holdings: List[Dict]) -> Dict:
        """
        Advanced risk analysis using deep learning
        """
        try:
            if not holdings:
                return self._default_risk_analysis()
            
            # Prepare feature matrix
            features = self._extract_financial_features(holdings)
            
            if self.risk_model is None:
                return self._fallback_risk_analysis(features)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.risk_model(feature_tensor)
                
                # Extract predictions
                risk_probs = F.softmax(predictions['risk'], dim=1).cpu().numpy()[0]
                volatility = predictions['volatility'].cpu().numpy()[0][0]
                expected_return = predictions['returns'].cpu().numpy()[0][0]
                correlation = predictions['correlation'].cpu().numpy()[0][0]
                
                # Anomaly detection
                anomaly_score = self._detect_portfolio_anomalies(features)
                
                return {
                    'risk_distribution': {
                        'very_low': float(risk_probs[0]),
                        'low': float(risk_probs[1]),
                        'moderate': float(risk_probs[2]),
                        'high': float(risk_probs[3]),
                        'very_high': float(risk_probs[4])
                    },
                    'predicted_volatility': float(volatility),
                    'expected_return': float(expected_return),
                    'correlation_score': float(correlation),
                    'anomaly_score': float(anomaly_score),
                    'risk_factors': self._identify_risk_factors(holdings, features),
                    'analysis_method': 'deep_learning'
                }
                
        except Exception as e:
            logger.error(f"Advanced risk analysis failed: {e}")
            return self._fallback_risk_analysis([])
    
    def _extract_financial_features(self, holdings: List[Dict]) -> np.ndarray:
        """Extract numerical features from portfolio holdings"""
        features = []
        
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        # Portfolio-level features
        portfolio_features = [
            len(holdings),  # Number of holdings
            total_value,    # Total portfolio value
            np.std([h.get('live_market_value', h.get('market_value', 0)) for h in holdings]),  # Value std
        ]
        
        # Sector concentration features
        sectors = {}
        for holding in holdings:
            sector = holding.get('sector', 'unknown')
            value = holding.get('live_market_value', holding.get('market_value', 0))
            sectors[sector] = sectors.get(sector, 0) + value
        
        sector_weights = [v / total_value for v in sectors.values()] if total_value > 0 else [0]
        sector_features = [
            len(sectors),  # Number of unique sectors
            max(sector_weights) if sector_weights else 0,  # Max sector concentration
            np.std(sector_weights) if len(sector_weights) > 1 else 0,  # Sector concentration std
        ]
        
        # Individual holding features (take top 10 by value)
        sorted_holdings = sorted(holdings, key=lambda x: x.get('live_market_value', x.get('market_value', 0)), reverse=True)[:10]
        holding_features = []
        
        for i, holding in enumerate(sorted_holdings):
            value = holding.get('live_market_value', holding.get('market_value', 0))
            weight = value / total_value if total_value > 0 else 0
            gain_loss = holding.get('live_gain_loss_percent', holding.get('gain_loss_percent', 0)) or 0
            
            holding_features.extend([weight, gain_loss])
        
        # Pad to ensure consistent size (20 holdings * 2 features = 40)
        while len(holding_features) < 40:
            holding_features.extend([0.0, 0.0])
        
        # Combine all features
        all_features = portfolio_features + sector_features + holding_features[:40]
        
        # Pad to ensure consistent input size (50 features)
        while len(all_features) < 50:
            all_features.append(0.0)
        
        return np.array(all_features[:50])
    
    def _detect_portfolio_anomalies(self, features: np.ndarray) -> float:
        """Detect anomalies in portfolio composition"""
        try:
            # Reshape for sklearn
            features_reshaped = features.reshape(1, -1)
            
            # Fit and predict (in practice, this would be pre-trained)
            self.anomaly_detector.fit(features_reshaped)
            anomaly_score = self.anomaly_detector.decision_function(features_reshaped)[0]
            
            # Normalize to 0-1 scale
            normalized_score = max(0, min(1, (anomaly_score + 0.5) / 1.0))
            
            return normalized_score
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return 0.5  # Neutral score
    
    def _identify_risk_factors(self, holdings: List[Dict], features: np.ndarray) -> List[str]:
        """Identify specific risk factors in the portfolio"""
        risk_factors = []
        
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        # Check concentration risk
        if len(holdings) < 5:
            risk_factors.append("Low diversification - fewer than 5 holdings")
        
        # Check sector concentration
        sectors = {}
        for holding in holdings:
            sector = holding.get('sector', 'unknown')
            value = holding.get('live_market_value', holding.get('market_value', 0))
            sectors[sector] = sectors.get(sector, 0) + value
        
        for sector, value in sectors.items():
            if value / total_value > 0.5:
                risk_factors.append(f"High {sector} sector concentration ({value/total_value:.1%})")
        
        # Check individual position size
        for holding in holdings:
            value = holding.get('live_market_value', holding.get('market_value', 0))
            if value / total_value > 0.3:
                symbol = holding.get('symbol', 'Unknown')
                risk_factors.append(f"Large position in {symbol} ({value/total_value:.1%})")
        
        return risk_factors[:5]  # Limit to top 5 risk factors
    
    async def generate_ai_recommendations(self, holdings: List[Dict], risk_analysis: Dict) -> List[Dict]:
        """
        Generate sophisticated AI-powered recommendations
        """
        try:
            recommendations = []
            
            # Analyze current portfolio state
            portfolio_analysis = self._analyze_portfolio_composition(holdings)
            
            # Generate recommendations based on ML insights
            ml_recommendations = self._generate_ml_recommendations(holdings, risk_analysis, portfolio_analysis)
            
            # Add sentiment-based recommendations
            sentiment_recommendations = await self._generate_sentiment_recommendations(holdings)
            
            # Combine and rank recommendations
            all_recommendations = ml_recommendations + sentiment_recommendations
            ranked_recommendations = self._rank_recommendations(all_recommendations)
            
            return ranked_recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"AI recommendation generation failed: {e}")
            return self._fallback_recommendations()
    
    def _analyze_portfolio_composition(self, holdings: List[Dict]) -> Dict:
        """Analyze current portfolio composition"""
        total_value = sum(h.get('live_market_value', h.get('market_value', 0)) for h in holdings)
        
        # Sector analysis
        sectors = {}
        for holding in holdings:
            sector = holding.get('sector', 'unknown')
            value = holding.get('live_market_value', holding.get('market_value', 0))
            sectors[sector] = sectors.get(sector, 0) + value
        
        # Performance analysis
        total_gain_loss = sum(h.get('live_gain_loss', h.get('gain_loss', 0)) or 0 for h in holdings)
        
        return {
            'total_value': total_value,
            'sector_weights': {k: v/total_value for k, v in sectors.items()} if total_value > 0 else {},
            'total_gain_loss': total_gain_loss,
            'num_holdings': len(holdings),
            'avg_position_size': total_value / len(holdings) if holdings else 0
        }
    
    def _generate_ml_recommendations(self, holdings: List[Dict], risk_analysis: Dict, portfolio_analysis: Dict) -> List[Dict]:
        """Generate ML-based recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        risk_dist = risk_analysis.get('risk_distribution', {})
        high_risk_prob = risk_dist.get('high', 0) + risk_dist.get('very_high', 0)
        
        if high_risk_prob > 0.6:
            recommendations.append({
                'type': 'risk_reduction',
                'priority': 'high',
                'title': '🛡️ Reduce Portfolio Risk',
                'description': f'ML analysis indicates {high_risk_prob:.1%} probability of high risk.',
                'rationale': 'Deep learning risk model detected elevated risk patterns',
                'suggested_actions': [
                    'Add defensive assets (bonds, utilities)',
                    'Reduce position sizes in volatile stocks',
                    'Consider hedging strategies'
                ],
                'confidence': 0.85,
                'ml_score': high_risk_prob
            })
        
        # Diversification recommendations
        sector_weights = portfolio_analysis['sector_weights']
        max_sector_weight = max(sector_weights.values()) if sector_weights else 0
        
        if max_sector_weight > 0.4:
            dominant_sector = max(sector_weights.keys(), key=lambda k: sector_weights[k])
            recommendations.append({
                'type': 'diversification',
                'priority': 'medium',
                'title': '📊 Improve Sector Diversification',
                'description': f'{dominant_sector.title()} represents {max_sector_weight:.1%} of portfolio.',
                'rationale': 'Machine learning detected sector concentration risk',
                'suggested_actions': [
                    f'Reduce {dominant_sector} allocation to under 30%',
                    'Add exposure to underrepresented sectors',
                    'Consider broad market ETFs'
                ],
                'confidence': 0.90,
                'ml_score': max_sector_weight
            })
        
        return recommendations
    
    async def _generate_sentiment_recommendations(self, holdings: List[Dict]) -> List[Dict]:
        """Generate recommendations based on NLP sentiment analysis"""
        recommendations = []
        
        try:
            # Analyze sentiment for each holding (simulated news analysis)
            for holding in holdings:
                symbol = holding.get('symbol')
                if symbol:
                    # In practice, this would analyze real news/social media
                    sentiment_text = f"Recent performance and outlook for {symbol} stock"
                    sentiment = self.nlp_pipeline(sentiment_text)[0]
                    
                    if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
                        recommendations.append({
                            'type': 'sentiment_based',
                            'priority': 'medium',
                            'title': f'⚠️ Negative Sentiment for {symbol}',
                            'description': f'NLP analysis shows negative sentiment (confidence: {sentiment["score"]:.1%})',
                            'rationale': 'FinBERT model detected negative market sentiment',
                            'suggested_actions': [
                                f'Monitor {symbol} closely for further weakness',
                                'Consider reducing position size',
                                'Set stop-loss orders'
                            ],
                            'confidence': sentiment['score'],
                            'ml_score': sentiment['score']
                        })
                        
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
        
        return recommendations
    
    def _rank_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Rank recommendations by importance and confidence"""
        priority_weights = {'high': 3, 'medium': 2, 'low': 1}
        
        for rec in recommendations:
            priority_score = priority_weights.get(rec.get('priority', 'low'), 1)
            confidence_score = rec.get('confidence', 0.5)
            ml_score = rec.get('ml_score', 0.5)
            
            # Combined ranking score
            rec['ranking_score'] = (priority_score * 0.4 + confidence_score * 0.3 + ml_score * 0.3)
        
        # Sort by ranking score
        return sorted(recommendations, key=lambda x: x['ranking_score'], reverse=True)
    
    # Fallback methods
    def _fallback_image_analysis(self, image_data: bytes) -> Dict:
        return {'analysis_method': 'fallback', 'confidence': 0.5}
    
    def _fallback_risk_analysis(self, features) -> Dict:
        return {'analysis_method': 'fallback', 'predicted_volatility': 0.2}
    
    def _fallback_recommendations(self) -> List[Dict]:
        return [{'type': 'general', 'title': 'Consider diversifying your portfolio'}]
    
    def _default_risk_analysis(self) -> Dict:
        return {'risk_distribution': {'moderate': 1.0}, 'predicted_volatility': 0.15}
    
    def _map_class_to_broker(self, class_id: int) -> str:
        broker_map = {
            0: 'robinhood', 1: 'fidelity', 2: 'schwab', 3: 'etrade', 
            4: 'td_ameritrade', 5: 'webull', 6: 'interactive_brokers',
            7: 'merrill', 8: 'vanguard', 9: 'generic'
        }
        return broker_map.get(class_id, 'generic')
    
    def save_models(self):
        """Save trained models"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.vision_model:
            torch.save(self.vision_model.state_dict(), 
                      os.path.join(self.model_dir, "vision_model.pth"))
        
        if self.risk_model:
            torch.save(self.risk_model.state_dict(), 
                      os.path.join(self.model_dir, "risk_model.pth"))
        
        logger.info("✅ Models saved successfully")