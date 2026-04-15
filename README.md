# AI Portfolio Analyzer

![Project Banner](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 1. Project Title and Description

**AI Portfolio Analyzer** is an advanced, AI-powered solution designed to analyze investment portfolios. It leverages cutting-edge machine learning and deep learning techniques to provide comprehensive insights, real-time market data integration, and predictive analytics. A unique feature of this system is its ability to process portfolio screenshots, extracting holdings information using computer vision, and then performing a detailed financial analysis. This tool aims to empower investors with deeper understanding of their portfolio's risk, diversification, and performance, along with actionable recommendations.

## 2. Features

The AI Portfolio Analyzer offers a robust set of features to provide a holistic view of your investments:

*   **Screenshot-based Portfolio Extraction:** Utilize deep learning models to automatically identify and extract investment holdings from uploaded portfolio screenshots.
*   **Enhanced Risk Analysis:**
    *   Calculates traditional metrics like **Beta, Sharpe Ratio, Volatility, Value at Risk (VaR 95%), and Maximum Drawdown**.
    *   Provides an **ML-based risk distribution**, **predicted volatility**, **expected return**, **correlation score**, and **anomaly detection**.
    *   Assigns an overall **Risk Level** (e.g., Moderate) and **Risk Score**.
*   **Comprehensive Diversification Analysis:**
    *   Analyzes **sector, asset class, and geographic diversification**.
    *   Leverages ML to assess **inter-asset correlations** for true diversification insights.
*   **Advanced Performance Metrics:**
    *   Reports **Total Return, Annualized Return, Year-to-Date (YTD) Return, and Monthly Returns**.
    *   Compares performance against benchmarks (**Benchmark Comparison, Alpha, Tracking Error**).
    *   Includes **ML-based performance predictions**, **risk-adjusted returns**, **performance consistency**, and **market timing scores**.
*   **Intelligent Recommendations:** Generates actionable, AI-driven recommendations to optimize portfolio structure, manage risk, and improve potential returns.
*   **Real-time Market Data Integration:** Incorporates up-to-date market data to ensure analyses and predictions are relevant and timely.
*   **Robustness & Fallback:** Includes a fallback mechanism to provide basic analysis even if advanced ML/DL components encounter issues, ensuring continuous service.
*   **Detailed Insights:** Provides an overall score, summary, and advanced insights derived from the deep learning analysis.
*   **Asynchronous Processing:** Built with `asyncio` for efficient, non-blocking operations, suitable for high-throughput environments.

## 3. Installation

To set up the AI Portfolio Analyzer, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Malav2002/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The project relies on several Python libraries for data manipulation, machine learning, deep learning, and asynchronous operations.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is assumed to exist in the root or `ml-service` directory. If not, you'll need to create one with `numpy`, `pandas`, `torch`, `scikit-learn`, `transformers`, `fastapi` (if it's a web service), `uvicorn`, `Pillow`, `opencv-python`, etc.)*

4.  **Download ML Models (if applicable):**
    Some deep learning models might require pre-trained weights or specific configurations.
    *   Check for a `models/` directory or specific instructions within the `ml-service` for model downloads or setup.
    *   *(Example placeholder: `python scripts/download_models.py`)*

## 4. Usage

The AI Portfolio Analyzer is designed to be consumed as a service, likely via an API. The core functionality involves sending an image of a portfolio and optionally, a list of known holdings, to receive a comprehensive analysis.

### Running the Service (Example with FastAPI/Uvicorn)

Assuming this is exposed as a web service (e.g., using FastAPI):

1.  **Start the service:**
    ```bash
    cd ml-service
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    *(Note: `main:app` is a placeholder for your FastAPI application entry point.)*

### Example API Call (Python)

You would typically interact with the service using an HTTP client.

```python
import httpx
import asyncio
import json

async def analyze_my_portfolio(image_path: str, known_holdings: list = None):
    url = "http://localhost:8000/analyze-portfolio-image" # Example endpoint
    
    # Prepare image data
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Prepare holdings data (optional)
    if known_holdings is None:
        known_holdings = [] # Send empty list if no known holdings

    files = {'image': ('portfolio_screenshot.png', image_data, 'image/png')}
    data = {'holdings': json.dumps(known_holdings)} # Holdings sent as JSON string

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, files=files, data=data)
            response.raise_for_status() # Raise an exception for HTTP errors
            
            analysis_result = response.json()
            print("Portfolio Analysis Result:")
            print(json.dumps(analysis_result, indent=2))
            return analysis_result
            
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage:
    # Replace 'path/to/your/screenshot.png' with an actual image file
    # Replace with your actual holdings data if you want to provide it
    sample_holdings = [
        {"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0, "currency": "USD"},
        {"symbol": "MSFT", "quantity": 5, "purchase_price": 250.0, "currency": "USD"}
    ]
    
    # asyncio.run(analyze_my_portfolio("path/to/your/screenshot.png", sample_holdings))
    print("Please provide a valid image path and uncomment the asyncio.run line to test.")
```

## 5. API Reference

The core functionality is exposed through a primary asynchronous function, which would typically be wrapped by a web API endpoint.

### `POST /analyze-portfolio-image` (Example Endpoint)

**Description:** Performs an enhanced portfolio analysis, including deep learning-based image processing to extract holdings and comprehensive financial metric calculation.

**Request Body:**
*   **`image`** (File): The portfolio screenshot image data (e.g., PNG, JPEG).
*   **`holdings`** (JSON String, optional): A JSON string representing a list of known portfolio holdings. This can supplement or verify holdings extracted from the image.
    ```json
    [
      {"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0, "currency": "USD"},
      {"symbol": "MSFT", "quantity": 5, "purchase_price": 250.0, "currency": "USD"},
      // ... more holdings
    ]
    ```

**Response Body (`EnhancedPortfolioAnalysis`):**

```json
{
  "risk_metrics": {
    "portfolio_beta": 1.0,
    "sharpe_ratio": 0.5,
    "volatility": 0.2,
    "var_95": 0.0,
    "max_drawdown": 0.1,
    "risk_level": "MODERATE",
    "risk_score": 50,
    "ml_risk_distribution": {},
    "predicted_volatility": 0.2,
    "expected_return": 0.08,
    "correlation_score": 0.5,
    "anomaly_score": 0.1,
    "risk_factors": [],
    "confidence_level": 0.5
  },
  "diversification": {
    "sector_distribution": {},
    "asset_class_distribution": {},
    "geographic_distribution": {},
    "ml_correlation_matrix": {},
    "diversification_score": 75,
    "concentration_risk": 0.2
  },
  "performance": {
    "total_return": 0.0,
    "annualized_return": 0.0,
    "ytd_return": 0.0,
    "monthly_returns": [],
    "benchmark_comparison": 0.0,
    "alpha": 0.0,
    "tracking_error": 0.0,
    "performance_score": 50,
    "ml_performance_prediction": 0.0,
    "risk_adjusted_returns": 0.0,
    "performance_consistency": 0.5,
    "market_timing_score": 0.5
  },
  "recommendations": [
    {
      "type": "REBALANCE",
      "description": "Consider rebalancing your portfolio to reduce sector concentration in technology.",
      "impact": "MEDIUM",
      "suggested_action": {"sell": [{"symbol": "NVDA", "amount": "10%"}], "buy": [{"symbol": "JPM", "amount": "5%"}, {"symbol": "XOM", "amount": "5%"}]}
    }
  ],
  "overall_score": 70,
  "summary": "Your portfolio shows moderate risk with good diversification, but could benefit from rebalancing to optimize returns.",
  "analysis_timestamp": "2023-10-27T10:30:00Z",
  "ml_analysis_available": true,
  "image_analysis": {
    "extracted_holdings": [
      {"symbol": "AAPL", "quantity": 10},
      {"symbol": "GOOG", "quantity": 5}
    ],
    "confidence": 0.95
  },
  "advanced_insights": {
    "sentiment_analysis_on_news": "Neutral for tech sector",
    "macroeconomic_impact": "Inflationary pressures could affect growth stocks."
  }
}
```

### Other Internal API Functions (if exposed directly or via other endpoints):

*   **`GET /analyze-risk`**: Provides detailed risk analysis for a given portfolio.
    *   **Request Body:** `{"holdings": [...]}`
    *   **Response:** `Dict` containing risk metrics.
*   **`GET /analyze-performance`**: Analyzes historical performance data.
    *   **Request Body:** `{"performance_data": {...}}`
    *   **Response:** `Dict` containing performance metrics.

## 6. Configuration

The AI Portfolio Analyzer can be configured using environment variables or a configuration file (e.g., `config.ini` or `settings.py`).

**Common Configuration Parameters:**

*   **`LOG_LEVEL`**: (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`) Controls the verbosity of logging.
*   **`MODEL_PATH_IMAGE_RECOGNITION`**: Path to the pre-trained deep learning model for image analysis.
*   **`MODEL_PATH_RISK_PREDICTOR`**: Path to the ML model for risk prediction.
*   **`MARKET_DATA_API_KEY`**: API key for real-time market data providers.
*   **`MARKET_DATA_BASE_URL`**: Base URL for the market data API.
*   **`FALLBACK_ENABLED`**: (e.g., `True`/`False`) Whether to enable the basic analysis fallback if enhanced analysis fails.
*   **`ASYNC_TIMEOUT_SECONDS`**: Timeout for asynchronous operations (e.g., API calls to external services).

**Example `.env` file:**

```
LOG_LEVEL=INFO
MODEL_PATH_IMAGE_RECOGNITION=./models/image_extractor_v2.pth
MODEL_PATH_RISK_PREDICTOR=./models/risk_predictor_v1.pkl
MARKET_DATA_API_KEY=YOUR_MARKET_DATA_API_KEY
MARKET_DATA_BASE_URL=https://api.marketdata.com/v1
FALLBACK_ENABLED=True
ASYNC_TIMEOUT_SECONDS=30
```

## 7. Contributing

We welcome contributions to the AI Portfolio Analyzer! If you're interested in improving the project, please follow these guidelines:

1.  **Fork the repository.**
2.  **Clone your forked repository:**
    ```bash
    git clone https://github.com/YourUsername/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer
    ```
3.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
    or
    ```bash
    git checkout -b bugfix/issue-description
    ```
4.  **Make your changes.** Ensure your code adheres to the project's coding style (e.g., PEP 8).
5.  **Write and run tests** to ensure your changes work as expected and don't introduce regressions.
6.  **Commit your changes** with a clear and descriptive commit message:
    ```bash
    git commit -m "feat: Add new feature for X"
    ```
    or
    ```bash
    git commit -m "fix: Resolve bug in Y calculation"
    ```
7.  **Push your branch** to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
8.  **Open a Pull Request** to the `main` branch of the original repository. Provide a detailed description of your changes.

## 8. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2023 Malav2002

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```