This document provides comprehensive API documentation for the `Malav2002/ai-portfolio-analyzer` project, an AI-powered service designed to analyze investment portfolios, including capabilities for processing portfolio screenshots and providing real-time market insights and predictions.

---

# AI Portfolio Analyzer API Documentation

## 1. Overview and Purpose

The AI Portfolio Analyzer API provides advanced capabilities for in-depth investment portfolio analysis. Leveraging machine learning and deep learning models, it offers features such as enhanced risk assessment, performance evaluation, diversification analysis, and personalized recommendations. A unique aspect of this API is its ability to process portfolio screenshots, extracting relevant data for analysis, thereby simplifying the input process for users. The API aims to provide actionable insights to optimize investment strategies and manage risk effectively.

## 2. Functions and Classes

This section details the primary functions and data structures exposed by the API.

### Classes

#### `EnhancedPortfolioAnalysis`

A data class representing the comprehensive output of an enhanced portfolio analysis, combining various metrics, recommendations, and ML-derived insights.

```python
@dataclass
class EnhancedPortfolioAnalysis:
    risk_metrics: EnhancedRiskMetrics
    diversification: EnhancedDiversificationMetrics
    performance: EnhancedPerformanceMetrics
    recommendations: List[EnhancedRecommendation]
    overall_score: int
    summary: str
    analysis_timestamp: str
    ml_analysis_available: bool
    image_analysis: Dict
    advanced_insights: Dict
```

**Attributes:**

*   **`risk_metrics`**: (`EnhancedRiskMetrics`) Detailed metrics related to portfolio risk.
*   **`diversification`**: (`EnhancedDiversificationMetrics`) Metrics assessing the diversification quality of the portfolio.
*   **`performance`**: (`EnhancedPerformanceMetrics`) Comprehensive performance indicators.
*   **`recommendations`**: (`List[EnhancedRecommendation]`) A list of AI-generated recommendations for portfolio improvement.
*   **`overall_score`**: (`int`) An aggregated score reflecting the overall health and potential of the portfolio (0-100).
*   **`summary`**: (`str`) A textual summary of the analysis findings.
*   **`analysis_timestamp`**: (`str`) The UTC timestamp when the analysis was performed.
*   **`ml_analysis_available`**: (`bool`) Indicates if advanced ML/DL analysis was successfully applied.
*   **`image_analysis`**: (`Dict`) Contains results specifically from image processing, such as detected holdings, confidence scores, and visual insights.
*   **`advanced_insights`**: (`Dict`) Additional deep learning-derived insights and predictions not covered by other metrics.

### Asynchronous Functions

#### `analyze_portfolio_with_image`

Performs an enhanced portfolio analysis, including deep learning-based image processing to extract holdings from a screenshot, combined with traditional and ML-driven financial analysis.

```python
async def analyze_portfolio_with_image(self, image_data: bytes, holdings: List[Dict]) -> EnhancedPortfolioAnalysis:
```

#### `analyze_risk`

Conducts a detailed risk analysis of the provided portfolio data using AI models to predict risk levels, identify anomalies, and assess various risk factors.

```python
async def analyze_risk(self, portfolio_data: Dict) -> Dict:
```

#### `analyze_performance`

Evaluates the historical and projected performance of a portfolio, providing key metrics such as returns, volatility, and risk-adjusted returns.

```python
async def analyze_performance(self, performance_data: Dict) -> Dict:
```

## 3. Parameters

This section details the parameters for each public API function.

### `analyze_portfolio_with_image`

*   **`image_data`**
    *   **Type**: `bytes`
    *   **Description**: Raw byte data of the portfolio screenshot image (e.g., PNG, JPEG).
    *   **Required**: Yes
*   **`holdings`**
    *   **Type**: `List[Dict]`
    *   **Description**: An optional list of known portfolio holdings. This can supplement or override holdings detected from the image. Each dictionary in the list should represent a single asset with at least `symbol` and `quantity`.
    *   **Required**: Yes (can be an empty list if relying solely on image detection)
    *   **Example `Dict` Structure**:
        ```json
        {
            "symbol": "AAPL",
            "quantity": 10,
            "purchase_price": 150.00,
            "current_price": 175.50,
            "sector": "Technology",
            "currency": "USD"
        }
        ```
        *(Note: The API will attempt to infer missing fields if not provided, but providing more data improves accuracy.)*

### `analyze_risk`

*   **`portfolio_data`**
    *   **Type**: `Dict`
    *   **Description**: A dictionary containing the portfolio's current state, primarily its holdings.
    *   **Required**: Yes
    *   **Example `Dict` Structure**:
        ```json
        {
            "holdings": [
                {
                    "symbol": "GOOGL",
                    "quantity": 5,
                    "purchase_price": 1000.00,
                    "current_price": 1200.00,
                    "sector": "Technology"
                },
                {
                    "symbol": "MSFT",
                    "quantity": 15,
                    "purchase_price": 200.00,
                    "current_price": 250.00,
                    "sector": "Technology"
                }
            ],
            "cash_balance": 5000.00,
            "portfolio_value": 20000.00
        }
        ```
        *(The `holdings` key is critical for risk analysis.)*

### `analyze_performance`

*   **`performance_data`**
    *   **Type**: `Dict`
    *   **Description**: A dictionary containing historical performance data or current portfolio state necessary for performance calculations.
    *   **Required**: Yes
    *   **Example `Dict` Structure**:
        ```json
        {
            "holdings": [
                {
                    "symbol": "AMZN",
                    "quantity": 2,
                    "purchase_price": 100.00,
                    "current_price": 130.00,
                    "historical_prices": {
                        "2023-01-01": 90.00,
                        "2023-02-01": 105.00,
                        "2023-03-01": 115.00
                    }
                }
            ],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        ```
        *(The specific keys required within `performance_data` will depend on the depth of performance analysis desired, but `holdings` with historical price data is a common requirement.)*

## 4. Return Values

This section describes the return values for each public API function.

### `analyze_portfolio_with_image`

*   **Type**: `EnhancedPortfolioAnalysis`
*   **Description**: A comprehensive object containing all calculated metrics, ML-derived insights, and recommendations for the analyzed portfolio. Refer to the `EnhancedPortfolioAnalysis` class definition for detailed attributes.

### `analyze_risk`

*   **Type**: `Dict`
*   **Description**: A dictionary containing various risk metrics and insights.
*   **Example `Dict` Structure**:
    ```json
    {
        "risk_score": 65,
        "risk_level": "MODERATE_HIGH",
        "volatility": 0.18,
        "sharpe_ratio": 0.85,
        "var_95": 0.03,
        "max_drawdown": 0.12,
        "predicted_volatility": 0.20,
        "expected_return": 0.10,
        "correlation_score": 0.6,
        "anomaly_detection": {
            "anomalies_found": true,
            "details": [
                {"holding": "XYZ", "type": "unusual_price_movement", "score": 0.9}
            ]
        },
        "risk_factors": ["market_risk", "concentration_risk"],
        "ml_risk_distribution": {
            "low": 0.1, "moderate": 0.4, "high": 0.5
        }
    }
    ```
    *(Note: If no holdings are provided or analysis fails, an empty/default risk analysis dictionary will be returned.)*

### `analyze_performance`

*   **Type**: `Dict`
*   **Description**: A dictionary containing key performance indicators for the portfolio.
*   **Example `Dict` Structure**:
    ```json
    {
        "total_return": 0.08,
        "annualized_return": 0.10,
        "volatility": 0.15,
        "sharpe_ratio": 0.67,
        "alpha": 0.02,
        "beta": 1.1,
        "monthly_returns": [0.01, -0.005, 0.02, ...],
        "benchmark_comparison": 0.015,
        "performance_score": 75,
        "ml_performance_prediction": 0.12,
        "risk_adjusted_returns": 0.07,
        "performance_consistency": 0.7
    }
    ```

## 5. Usage Examples

Here are practical Python code examples demonstrating how to interact with the API.

Assume `analyzer_service` is an instantiated client for the `AIPortfolioAnalyzer` or `EnhancedAIPortfolioAnalyzer` class.

### Example 1: Enhanced Portfolio Analysis with Image and Known Holdings

```python
import asyncio
import base64
from typing import List, Dict

# Assume EnhancedAIPortfolioAnalyzer is properly initialized
# from ml-service.src.services.enhanced_ai_analyzer import EnhancedAIPortfolioAnalyzer
# analyzer_service = EnhancedAIPortfolioAnalyzer(...)

async def run_enhanced_analysis():
    # Load image data (replace with actual image loading)
    try:
        with open("path/to/your/portfolio_screenshot.png", "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        print("Error: Portfolio screenshot not found. Using dummy image data.")
        # Create dummy image data for demonstration if file not found
        image_bytes = b"dummy_image_data_bytes" 

    # Provide some known holdings to supplement image analysis
    known_holdings: List[Dict] = [
        {
            "symbol": "TSLA",
            "quantity": 5,
            "purchase_price": 200.00,
            "current_price": 250.00,
            "sector": "Automotive"
        },
        {
            "symbol": "NVDA",
            "quantity": 10,
            "purchase_price": 400.00,
            "current_price": 450.00,
            "sector": "Technology"
        }
    ]

    try:
        # Call the enhanced analysis function
        analysis_result = await analyzer_service.analyze_portfolio_with_image(
            image_data=image_bytes,
            holdings=known_holdings
        )

        print("\n--- Enhanced Portfolio Analysis Result ---")
        print(f"Overall Score: {analysis_result.overall_score}")
        print(f"Summary: {analysis_result.summary}")
        print(f"Risk Level: {analysis_result.risk_metrics.risk_level.name}")
        print(f"Total Return: {analysis_result.performance.total_return:.2f}%")
        print(f"ML Analysis Available: {analysis_result.ml_analysis_available}")
        print(f"Image Analysis Insights: {analysis_result.image_analysis.get('detected_items', 'N/A')}")
        print("\nRecommendations:")
        for rec in analysis_result.recommendations:
            print(f"- {rec.category}: {rec.description} (Priority: {rec.priority})")

    except Exception as e:
        print(f"An error occurred during enhanced analysis: {e}")

# To run this example:
# asyncio.run(run_enhanced_analysis())
```

### Example 2: Detailed Risk Analysis

```python
import asyncio
from typing import Dict

# Assume AIPortfolioAnalyzer is properly initialized
# from ml-service.src.services.ai_portfolio_analyzer import AIPortfolioAnalyzer
# analyzer_service = AIPortfolioAnalyzer(...)

async def run_risk_analysis():
    portfolio_data: Dict = {
        "holdings": [
            {
                "symbol": "AAPL",
                "quantity": 10,
                "purchase_price": 150.00,
                "current_price": 175.50,
                "sector": "Technology"
            },
            {
                "symbol": "GOOGL",
                "quantity": 5,
                "purchase_price": 1000.00,
                "current_price": 1200.00,
                "sector": "Technology"
            },
            {
                "symbol": "JPM",
                "quantity": 20,
                "purchase_price": 120.00,
                "current_price": 130.00,
                "sector": "Financials"
            }
        ],
        "cash_balance": 5000.00
    }

    try:
        risk_analysis_result = await analyzer_service.analyze_risk(portfolio_data)

        print("\n--- Risk Analysis Result ---")
        print(f"Risk Score: {risk_analysis_result.get('risk_score', 'N/A')}")
        print(f"Risk Level: {risk_analysis_result.get('risk_level', 'N/A')}")
        print(f"Volatility: {risk_analysis_result.get('volatility', 'N/A'):.2f}")
        print(f"Sharpe Ratio: {risk_analysis_result.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"Anomalies Found: {risk_analysis_result.get('anomaly_detection', {}).get('anomalies_found', False)}")
        if risk_analysis_result.get('anomaly_detection', {}).get('details'):
            print("Anomaly Details:")
            for anomaly in risk_analysis_result['anomaly_detection']['details']:
                print(f"  - Holding: {anomaly.get('holding')}, Type: {anomaly.get('type')}, Score: {anomaly.get('score'):.2f}")

    except Exception as e:
        print(f"An error occurred during risk analysis: {e}")

# To run this example:
# asyncio.run(run_risk_analysis())
```

### Example 3: Portfolio Performance Analysis

```python
import asyncio
from typing import Dict

# Assume AIPortfolioAnalyzer is properly initialized
# from ml-service.src.services.ai_portfolio_analyzer import AIPortfolioAnalyzer
# analyzer_service = AIPortfolioAnalyzer(...)

async def run_performance_analysis():
    performance_data: Dict = {
        "holdings": [
            {
                "symbol": "MSFT",
                "quantity": 10,
                "purchase_price": 250.00,
                "current_price": 300.00,
                "historical_prices": {
                    "2023-01-01": 240.00,
                    "2023-02-01": 260.00,
                    "2023-03-01": 275.00,
                    "2023-04-01": 290.00,
                    "2023-05-01": 300.00
                }
            }
        ],
        "start_date": "2023-01-01",
        "end_date": "2023-05-01"
    }

    try:
        performance_result = await analyzer_service.analyze_performance(performance_data)

        print("\n--- Performance Analysis Result ---")
        print(f"Total Return: {performance_result.get('total_return', 'N/A'):.2f}")
        print(f"Annualized Return: {performance_result.get('annualized_return', 'N/A'):.2f}")
        print(f"Volatility: {performance_result.get('volatility', 'N/A'):.2f}")
        print(f"Sharpe Ratio: {performance_result.get('sharpe_ratio', 'N/A'):.2f}")
        print(f"Performance Score: {performance_result.get('performance_score', 'N/A')}")

    except Exception as e:
        print(f"An error occurred during performance analysis: {e}")

# To run this example:
# asyncio.run(run_performance_analysis())
```

## 6. Error Handling

The API is designed with robust error handling, catching various exceptions to ensure graceful degradation and informative feedback.

*   **`Exception`**: A general catch-all for unexpected errors during processing. This could include issues with ML model inference, data processing, or external service calls. When a general `Exception` occurs, the API will typically log the error and may return a default or fallback analysis result (e.g., `_fallback_to_basic_analysis` for `analyze_portfolio_with_image`) or an empty/partial dictionary for other functions, along with an error message.
*   **`ImportError`**: Indicates a missing dependency. While this should ideally be caught during deployment/setup, it can occur if the environment is misconfigured.
*   **Specific Errors**:
    *   **`FileNotFoundError`**: If the API attempts to load a local resource (e.g., a model file) that does not exist.
    *   **`ValueError` / `TypeError`**: May occur if input data (`holdings`, `portfolio_data`, `performance_data`) does not conform to the expected structure or types.
    *   **Image Processing Errors**: If `image_data` is corrupted, not a valid image format, or if the OCR/deep learning models fail to extract meaningful data. In such cases, the `image_analysis` field in `EnhancedPortfolioAnalysis` might indicate failure, and the analysis might proceed with only provided `holdings` or fall back to a basic analysis.
    *   **External API Errors**: If the service relies on external market data APIs and they fail or return unexpected data.

**General Error Response Pattern:**

In case of a critical error, functions like `analyze_portfolio_with_image` may return an `EnhancedPortfolioAnalysis` object where `ml_analysis_available` is `False`, `summary` contains an error message, and other metrics are set to default or zero values. Other functions like `analyze_risk` or `analyze_performance` might return a dictionary with an `error` key and a descriptive message, or a minimal/empty dictionary.

Clients should always check for the presence of expected data in the return objects and handle potential error messages or default values.

## 7. Common Use Cases

The AI Portfolio Analyzer API can be integrated into various applications and services to provide intelligent financial insights.

1.  **Personal Finance Management (PFM) Apps**:
    *   **Scenario**: A user wants to quickly get an overview of their investment portfolio without manually entering all holdings.
    *   **API Usage**: The PFM app calls `analyze_portfolio_with_image` with a screenshot of the user's brokerage account. The API extracts holdings, analyzes them, and returns a comprehensive `EnhancedPortfolioAnalysis` object, which the app then displays to the user, including risk scores, performance, and personalized recommendations.

2.  **Robo-Advisors**:
    *   **Scenario**: A robo-advisor platform needs to provide automated, data-driven investment advice and rebalancing suggestions.
    *   **API Usage**: The platform periodically calls `analyze_risk` and `analyze_performance` with the client's current portfolio data. Based on the returned risk metrics, performance indicators, and anomaly detection, the robo-advisor can identify if the portfolio deviates from the client's risk profile or investment goals and suggest adjustments.

3.  **Financial Planning Tools**:
    *   **Scenario**: A financial planner wants to quickly assess a client's existing portfolio for risk and diversification before a consultation.
    *   **API Usage**: The tool uses `analyze_risk` to get a detailed breakdown of the client's portfolio risk, including sector concentration, volatility, and correlation. It can also use `analyze_performance` to understand historical returns and risk-adjusted performance.

4.  **Market Research and Analytics Platforms**:
    *   **Scenario**: An institutional investor or analyst wants to analyze a large number of hypothetical portfolios or track market trends through aggregated portfolio data.
    *   **API Usage**: The platform can run `analyze_risk` and `analyze_performance` on various simulated portfolios to understand how different asset allocations or market conditions impact risk and return, aiding in strategy development.

5.  **Investment Education and Simulation**:
    *   **Scenario**: An educational platform wants to allow users to upload their portfolio screenshots or input hypothetical holdings to learn about investment analysis.
    *   **API Usage**: Users can leverage `analyze_portfolio_with_image` or provide `holdings` directly to receive detailed analysis, including explanations of risk metrics, diversification benefits, and performance drivers, helping them understand investment principles in a practical context.