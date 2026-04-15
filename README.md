# AI-Powered Portfolio Analyzer

![GitHub last commit](https://img.shields.io/github/last-commit/Malav2002/ai-portfolio-analyzer)
![GitHub stars](https://img.shields.io/github/stars/Malav2002/ai-portfolio-analyzer?style=social)
![GitHub issues](https://img.shields.io/github/issues/Malav2002/ai-portfolio-analyzer)
![License](https://img.shields.io/github/license/Malav2002/ai-portfolio-analyzer)

## 1. Project Title and Description

**AI-Powered Portfolio Analyzer** is an advanced financial intelligence platform designed to provide comprehensive analysis of investment portfolios. Leveraging cutting-edge Artificial Intelligence and Machine Learning techniques, this service goes beyond traditional portfolio analysis by incorporating real-time market data, predictive modeling, and even the ability to analyze portfolio screenshots using deep learning.

The core objective is to empower investors with deeper insights into their holdings, offering detailed risk assessments, performance evaluations, diversification metrics, and actionable recommendations to optimize their investment strategies. Whether you're a seasoned investor or just starting, this tool aims to provide clarity and foresight into your financial health.

## 2. Features

The AI-Powered Portfolio Analyzer is packed with a robust set of features to give you an unparalleled view of your investments:

*   **Intelligent Portfolio Analysis**:
    *   **Enhanced Risk Metrics**: Detailed assessment of portfolio risk including Beta, Sharpe Ratio, Volatility, Value at Risk (VaR), Maximum Drawdown, and an overall ML-driven risk score. Predicts future volatility and identifies key risk factors.
    *   **Diversification Analysis**: Evaluates portfolio diversification across sectors, asset classes, and geographies, providing insights into concentration risks and opportunities for better balance.
    *   **Performance Evaluation**: Calculates total, annualized, and YTD returns, benchmark comparisons, Alpha, Tracking Error, and a proprietary performance score. Includes ML-driven performance predictions and risk-adjusted returns.
*   **AI & Machine Learning Core**:
    *   **ML-Enhanced Holdings**: Utilizes machine learning to enrich raw portfolio holdings data with additional context and predictive attributes.
    *   **Predictive Analytics**: Forecasts future risk distributions, expected returns, and performance trends using advanced AI models.
    *   **Anomaly Detection**: Identifies unusual patterns or potential issues within the portfolio data that might indicate hidden risks or opportunities.
*   **Deep Learning Image Analysis**:
    *   **Screenshot Analysis**: Uniquely capable of processing portfolio screenshots (e.g., from brokerage apps) to extract holdings and relevant financial data using deep learning models. This enables analysis even without direct API access to brokerage accounts.
*   **Real-time Market Data Integration**: Incorporates up-to-the-minute market data to ensure analyses and predictions are based on the latest available information.
*   **Actionable Recommendations**: Generates personalized, data-driven recommendations to improve portfolio performance, reduce risk, and enhance diversification.
*   **Comprehensive Output**: Provides a structured `EnhancedPortfolioAnalysis` object containing all computed metrics, insights, and recommendations, along with an overall score and summary.
*   **Robustness and Fallback**: Includes a fallback mechanism to provide basic analysis even if advanced AI/ML or image processing components encounter issues, ensuring continuous service.

## 3. Installation

To get started with the AI-Powered Portfolio Analyzer, follow these steps:

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Malav2002/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    The project relies on several powerful libraries for data manipulation, machine learning, and deep learning.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be present in the root of the repository, listing `numpy`, `pandas`, `torch`, `sklearn`, `transformers`, `asyncio`, `dataclasses`, `enum`, `logging`, `datetime`, `Pillow` (for image processing), and potentially others.)*

4.  **Model Setup (if applicable)**:
    Some AI/ML models might require pre-trained weights or specific configurations. Check the `ml-service/models/` directory or project documentation for instructions on downloading or training necessary models.

## 4. Usage

The core functionality is exposed through an asynchronous service. You can integrate it into your application or run it as a standalone service.

### Example: Analyzing a Portfolio with an Image

The primary entry point for comprehensive analysis, including image processing, is `analyze_portfolio_with_image`.

```python
import asyncio
import base64
from typing import List, Dict
from ml_service.src.services.enhanced_ai_analyzer import EnhancedAIPortfolioAnalyzer

async def run_analysis():
    analyzer = EnhancedAIPortfolioAnalyzer()

    # Example holdings (replace with actual data)
    sample_holdings: List[Dict] = [
        {"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0, "current_price": 170.0, "sector": "Technology"},
        {"symbol": "MSFT", "quantity": 5, "purchase_price": 250.0, "current_price": 300.0, "sector": "Technology"},
        {"symbol": "GOOGL", "quantity": 3, "purchase_price": 100.0, "current_price": 120.0, "sector": "Communication Services"},
        {"symbol": "JPM", "quantity": 8, "purchase_price": 120.0, "current_price": 130.0, "sector": "Financials"},
    ]

    # Load a dummy image (replace with actual image bytes from a screenshot)
    # For demonstration, let's create a tiny dummy image.
    # In a real scenario, you would read bytes from a file or an HTTP request.
    try:
        with open("path/to/your/portfolio_screenshot.png", "rb") as f:
            image_data_bytes = f.read()
    except FileNotFoundError:
        print("Dummy image file not found. Using placeholder bytes.")
        # Create a very small, valid PNG image header + minimal data
        # This is just for the code to run without error, not for actual analysis.
        image_data_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )


    print("Starting enhanced portfolio analysis...")
    analysis_result = await analyzer.analyze_portfolio_with_image(image_data_bytes, sample_holdings)

    print("\n--- Analysis Results ---")
    print(f"Overall Score: {analysis_result.overall_score}")
    print(f"Summary: {analysis_result.summary}")
    print("\nRisk Metrics:")
    print(f"  Risk Level: {analysis_result.risk_metrics.risk_level.name}")
    print(f"  Volatility: {analysis_result.risk_metrics.volatility:.2f}")
    print(f"  Sharpe Ratio: {analysis_result.risk_metrics.sharpe_ratio:.2f}")
    print("\nPerformance Metrics:")
    print(f"  Total Return: {analysis_result.performance.total_return:.2f}")
    print(f"  Annualized Return: {analysis_result.performance.annualized_return:.2f}")
    print(f"  ML Performance Prediction: {analysis_result.performance.ml_performance_prediction:.2f}")
    print("\nRecommendations:")
    for i, rec in enumerate(analysis_result.recommendations):
        print(f"  {i+1}. {rec.recommendation_type.name}: {rec.description} (Priority: {rec.priority})")
    
    if analysis_result.ml_analysis_available:
        print("\nML Analysis Available: Yes")
        print(f"Image Analysis Insights: {analysis_result.image_analysis.get('extracted_info', 'N/A')}")
        print(f"Advanced Insights: {analysis_result.advanced_insights}")
    else:
        print("\nML Analysis Available: No (Fallback to basic analysis)")


if __name__ == "__main__":
    asyncio.run(run_analysis())
```

### Running as an API Service

Typically, this `ml-service` would be exposed via a web framework (e.g., FastAPI, Flask) to handle HTTP requests.

```python
# Example using FastAPI (conceptual)
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict
# from ml_service.src.services.enhanced_ai_analyzer import EnhancedAIPortfolioAnalyzer, EnhancedPortfolioAnalysis

# app = FastAPI()
# analyzer = EnhancedAIPortfolioAnalyzer()

# class Holding(BaseModel):
#     symbol: str
#     quantity: int
#     purchase_price: float
#     current_price: float
#     sector: str = None

# @app.post("/analyze-portfolio/", response_model=EnhancedPortfolioAnalysis)
# async def analyze_portfolio_endpoint(
#     holdings: List[Holding],
#     portfolio_screenshot: UploadFile = File(None)
# ):
#     try:
#         holdings_dicts = [h.dict() for h in holdings]
#         image_data = await portfolio_screenshot.read() if portfolio_screenshot else None
        
#         if image_data:
#             result = await analyzer.analyze_portfolio_with_image(image_data, holdings_dicts)
#         else:
#             # Fallback to analysis without image if no image provided
#             result = await analyzer._fallback_to_basic_analysis(holdings_dicts) # Or a dedicated non-image analysis method
            
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# To run this (if implemented): `uvicorn main:app --reload`
```

## 5. API Reference

The core analysis logic resides in `ml-service/src/services/enhanced_ai_analyzer.py` and `ml-service/src/services/ai_portfolio_analyzer.py`.

### `EnhancedAIPortfolioAnalyzer` (from `enhanced_ai_analyzer.py`)

This class provides the most comprehensive analysis, including deep learning for image processing.

*   **`async analyze_portfolio_with_image(image_data: bytes, holdings: List[Dict]) -> EnhancedPortfolioAnalysis`**
    *   **Description**: The main entry point for enhanced portfolio analysis, incorporating deep learning for image processing of portfolio screenshots.
    *   **Parameters**:
        *   `image_data` (`bytes`): Raw bytes of the portfolio screenshot image (e.g., PNG, JPEG).
        *   `holdings` (`List[Dict]`): A list of dictionaries, where each dictionary represents a holding.
            *   Example holding format: `{"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0, "current_price": 170.0, "sector": "Technology"}`.
    *   **Returns**: `EnhancedPortfolioAnalysis` - A dataclass containing detailed risk, diversification, performance metrics, recommendations, and ML-specific insights.

### `AIPortfolioAnalyzer` (from `ai_portfolio_analyzer.py`)

This class provides core AI-driven analysis functionalities, which might be used internally by `EnhancedAIPortfolioAnalyzer` or exposed for specific tasks.

*   **`async analyze_risk(portfolio_data: Dict) -> Dict`**
    *   **Description**: Performs a detailed risk analysis using AI models, including risk prediction and anomaly detection.
    *   **Parameters**:
        *   `portfolio_data` (`Dict`): A dictionary containing portfolio information, typically with a `'holdings'` key.
    *   **Returns**: `Dict` - A dictionary containing various risk metrics and insights.

*   **`async analyze_performance(performance_data: Dict) -> Dict`**
    *   **Description**: Analyzes portfolio performance over time.
    *   **Parameters**:
        *   `performance_data` (`Dict`): A dictionary containing performance-related data.
    *   **Returns**: `Dict` - A dictionary with performance metrics like total return, annualized return, volatility, and Sharpe ratio.

### Key Data Structures

*   **`EnhancedPortfolioAnalysis`**:
    *   `risk_metrics`: `EnhancedRiskMetrics`
    *   `diversification`: `EnhancedDiversificationMetrics`
    *   `performance`: `EnhancedPerformanceMetrics`
    *   `recommendations`: `List[EnhancedRecommendation]`
    *   `overall_score`: `int`
    *   `summary`: `str`
    *   `analysis_timestamp`: `str`
    *   `ml_analysis_available`: `bool`
    *   `image_analysis`: `Dict` (e.g., extracted text, detected elements)
    *   `advanced_insights`: `Dict` (e.g., deep learning model outputs)

*   **`EnhancedRiskMetrics`**:
    *   `portfolio_beta`, `sharpe_ratio`, `volatility`, `var_95`, `max_drawdown`, `risk_level`, `risk_score`, `ml_risk_distribution`, `predicted_volatility`, `expected_return`, `correlation_score`, `anomaly_score`, `risk_factors`, `confidence_level`.

*   **`EnhancedPerformanceMetrics`**:
    *   `total_return`, `annualized_return`, `ytd_return`, `monthly_returns`, `benchmark_comparison`, `alpha`, `tracking_error`, `performance_score`, `ml_performance_prediction`, `risk_adjusted_returns`, `performance_consistency`, `market_timing_score`.

*   **`EnhancedDiversificationMetrics`**: (Not explicitly detailed in chunks, but implied)
    *   Would include metrics like sector concentration, asset class allocation, geographical distribution, etc.

*   **`EnhancedRecommendation`**:
    *   `recommendation_type`, `description`, `priority`, `suggested_action`, `impact_score`.

## 6. Configuration

The service can be configured using environment variables or a configuration file.

*   **Logging**: The `logger` global variable suggests that logging levels and output can be configured.
    *   `LOG_LEVEL`: Set to `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`).
*   **Model Paths**: Paths to pre-trained models for image analysis, risk prediction, etc.
    *   `ML_MODEL_PATH_RISK`: Path to the risk prediction model.
    *   `ML_MODEL_PATH_IMAGE_OCR`: Path to the OCR model for image processing.
    *   `ML_MODEL_PATH_IMAGE_DETECTION`: Path to the object detection model for image analysis.
*   **API Keys**: If integrating with external market data providers, API keys would be required.
    *   `MARKET_DATA_API_KEY`: API key for real-time market data.
*   **Performance Tuning**:
    *   `ASYNC_CONCURRENCY_LIMIT`: Limit for concurrent asynchronous operations.

Example `.env` file:
```
LOG_LEVEL=INFO
ML_MODEL_PATH_RISK=./models/risk_predictor.pt
MARKET_DATA_API_KEY=your_api_key_here
```

## 7. Contributing

We welcome contributions to the AI-Powered Portfolio Analyzer! If you have suggestions for improvements, new features, or bug fixes, please follow these guidelines:

1.  **Fork the repository**: Start by forking the `Malav2002/ai-portfolio-analyzer` repository to your GitHub account.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/YourUsername/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer
    ```
3.  **Create a new branch**:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    Or for bug fixes:
    ```bash
    git checkout -b bugfix/issue-description
    ```
4.  **Make your changes**: Implement your features or bug fixes. Ensure your code adheres to the existing style and conventions.
5.  **Write tests**: If applicable, add unit or integration tests for your changes.
6.  **Update documentation**: Ensure any new features or changes to existing ones are reflected in the documentation.
7.  **Commit your changes**:
    ```bash
    git commit -m "feat: Add new feature for X"
    ```
    (Use conventional commit messages like `feat:`, `fix:`, `docs:`, `refactor:`, etc.)
8.  **Push to your fork**:
    ```bash
    git push origin feature/your-feature-name
    ```
9.  **Create a Pull Request**: Go to the original repository on GitHub and open a new Pull Request from your branch. Provide a clear description of your changes and reference any relevant issues.

## 8. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

**Disclaimer**: This project is for informational and educational purposes only and should not be considered financial advice. Investment decisions should be made based on your own research and consultation with a qualified financial professional. The accuracy of predictions and analyses is not guaranteed.