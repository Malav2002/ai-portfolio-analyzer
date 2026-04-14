# AI-Powered Portfolio Analyzer

## 🚀 Project Title and Description

The **AI-Powered Portfolio Analyzer** is an advanced financial intelligence service designed to provide comprehensive and predictive analysis of investment portfolios. Leveraging cutting-edge Artificial Intelligence and Deep Learning techniques, this system can intelligently process portfolio screenshots, accurately extract individual holdings, integrate real-time market data, and generate sophisticated predictive insights. It offers investors a deeper, data-driven understanding of their portfolio's risk profile, diversification quality, and performance trajectory, moving beyond conventional analytical methods.

**Primary Language:** Python

## ✨ Features

*   **Screenshot-to-Portfolio Extraction:** Utilizes deep learning-based image analysis to automatically extract investment holdings and relevant data directly from portfolio screenshots (e.g., from brokerage platforms).
*   **Enhanced AI-Driven Metrics:** Provides a suite of advanced analytical metrics:
    *   **Risk Analysis:** Detailed assessment including portfolio beta, Sharpe ratio, volatility, Value at Risk (VaR), maximum drawdown, AI-predicted risk distributions, and anomaly detection.
    *   **Diversification Analysis:** Comprehensive evaluation of portfolio diversification across various dimensions (sectors, asset classes, geographies), offering ML-driven insights into optimal allocation strategies.
    *   **Performance Analysis:** In-depth performance metrics such as total return, annualized return, year-to-date (YTD) return, alpha, tracking error, and ML-predicted future performance. Includes sophisticated ML-based risk-adjusted returns.
*   **Real-time Market Data Integration:** Incorporates live market data feeds to ensure all analyses, predictions, and recommendations are based on the most current market conditions.
*   **Predictive Insights & Actionable Recommendations:** Generates intelligent, actionable recommendations for portfolio rebalancing, risk mitigation, and performance enhancement, all powered by sophisticated AI models.
*   **Robust Fallback Mechanism:** Includes a resilient fallback to basic analysis if the enhanced, deep learning-driven analysis encounters issues, ensuring continuous service availability and a baseline level of insight.
*   **Comprehensive Output Structure:** Delivers a structured `EnhancedPortfolioAnalysis` object containing all computed metrics, an overall score, a summary, analysis timestamp, and specific results from image processing and advanced deep learning insights.
*   **Scalable and Asynchronous Architecture:** Built using Python's `asyncio` for efficient handling of concurrent analysis requests, making it suitable for high-throughput environments.

## 🛠️ Installation

To set up the AI-Powered Portfolio Analyzer service locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Malav2002/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer/ml-service # Navigate to the service directory
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project relies on several Python libraries for data manipulation, machine learning (PyTorch, scikit-learn, Hugging Face Transformers), and asynchronous operations.
    ```bash
    pip install -r requirements.txt
    # If requirements.txt is not provided, you would typically install:
    # pip install numpy pandas torch scikit-learn transformers aiohttp Pillow python-dotenv
    ```
    *Note: `torch` and `transformers` can be substantial in size. Ensure you have sufficient disk space. If you plan to use GPU acceleration, install the appropriate `torch` version compatible with your CUDA setup.*

4.  **Model Setup (if applicable):**
    Pre-trained AI models for image analysis and financial predictions might need to be downloaded or configured. Check for a `models/` directory or specific instructions within the project for model initialization.

## 🚀 Usage

The core functionality of this service is designed to be exposed via an API endpoint, typically within a web service framework (e.g., FastAPI, Flask).

### Running the Service (Conceptual)

Assuming the service is implemented using a framework like FastAPI, you would start it using a WSGI server like Uvicorn:

```bash
# Example: If your main application file is `main.py` and the FastAPI app object is `app`
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Example API Call (Conceptual)

You would typically send a `POST` request to the service endpoint, providing the portfolio screenshot as binary data and the known holdings as JSON.

```python
import asyncio
import aiohttp # Or 'requests' for synchronous calls
import json

async def analyze_my_portfolio(service_url: str, image_path: str, holdings_data: list):
    """
    Sends a portfolio screenshot and holdings data to the AI analyzer service.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Prepare the request payload using multipart/form-data for file upload and JSON data
        data = aiohttp.FormData()
        data.add_field('image', image_bytes, filename='portfolio_screenshot.png', content_type='image/png')
        data.add_field('holdings', json.dumps(holdings_data), content_type='application/json')

        async with aiohttp.ClientSession() as session:
            async with session.post(service_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print("Portfolio Analysis Result:")
                    print(json.dumps(result, indent=2))
                    return result
                else:
                    error_text = await response.text()
                    print(f"Error during analysis: {response.status} - {error_text}")
                    return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    SERVICE_URL = "http://localhost:8000/analyze_portfolio_with_image" # Adjust if your service runs elsewhere
    SAMPLE_IMAGE_PATH = "path/to/your/portfolio_screenshot.png" # **IMPORTANT: Replace with an actual image path**

    # Example holdings data (can be an empty list if only image analysis is desired)
    sample_holdings = [
        {"symbol": "AAPL", "quantity": 10, "purchase_price": 150.0, "current_price": 170.0, "sector": "Technology"},
        {"symbol": "MSFT", "quantity": 5, "purchase_price": 250.0, "current_price": 300.0, "sector": "Technology"},
        {"symbol": "GOOG", "quantity": 2, "purchase_price": 100.0, "current_price": 120.0, "sector": "Communication Services"}
    ]

    print(f"To run this example, please ensure:")
    print(f"1. The AI-Powered Portfolio Analyzer service is running at {SERVICE_URL}.")
    print(f"2. You have replaced '{SAMPLE_IMAGE_PATH}' with the actual path to a portfolio screenshot image.")
    print(f"3. Then, uncomment the `asyncio.run` line below.")
    # asyncio.run(analyze_my_portfolio(SERVICE_URL, SAMPLE_IMAGE_PATH, sample_holdings))
```

## 📚 API Reference

The primary API endpoint for initiating an enhanced portfolio analysis is `analyze_portfolio_with_image`.

### `POST /analyze_portfolio_with_image`

Performs an enhanced AI-driven analysis of a portfolio, integrating data extracted from a screenshot with explicitly provided holdings.

*   **Method:** `POST`
*   **URL:** `/analyze_portfolio_with_image` (relative to your service's base URL)
*   **Content-Type:** `multipart/form-data` (recommended for file uploads)

#### Request Body Parameters:

*   `image`: (File/Binary)
    *   **Type:** `bytes` (binary image data, e.g., PNG, JPEG).
    *   **Description:** The screenshot of the user's investment portfolio. This image will be processed by deep learning models to identify and extract holdings.
    *   **Required:** Yes
*   `holdings`: (JSON String)
    *   **Type:** `List[Dict]`
    *   **Description:** A JSON string representing a list of dictionaries, where each dictionary details an individual holding in the portfolio. This data complements or validates information extracted from the image. Each holding dictionary should ideally contain:
        *   `symbol` (str): Stock ticker symbol (e.g., "AAPL").
        *   `quantity` (int/float): Number of shares held.
        *   `purchase_price` (float): Price at which the shares were purchased.
        *   `current_price` (float, optional): Current market price of the asset.
        *   `sector` (str, optional): Sector of the asset (e.g., "Technology").
        *   `asset_class` (str, optional): Asset class (e.g., "Equity", "Bond").
    *   **Required:** Yes (can be an empty list `[]` if analysis is solely based on image extraction, but providing data improves accuracy).

#### Response Body: `EnhancedPortfolioAnalysis` object

A JSON object detailing the comprehensive analysis of the portfolio.

```json
{
  "risk_metrics": {
    "portfolio_beta": float,           // Sensitivity to market movements
    "sharpe_ratio": float,             // Risk-adjusted return
    "volatility": float,               // Standard deviation of returns
    "var_95": float,                   // Value at Risk at 95% confidence
    "max_drawdown": float,             // Largest peak-to-trough decline
    "risk_level": "LOW" | "MODERATE" | "HIGH", // Categorized risk level
    "risk_score": int,                 // Overall risk score (0-100)
    "ml_risk_distribution": {},        // ML-predicted probability distribution of future risks
    "predicted_volatility": float,     // ML-predicted future volatility
    "expected_return": float,          // ML-predicted expected return
    "correlation_score": float,        // Average correlation within portfolio
    "anomaly_score": float,            // Score indicating unusual risk patterns
    "risk_factors": ["str"],           // List of identified key risk factors
    "confidence_level": float          // Confidence in risk assessment
  },
  "diversification": {
    "sector_diversification_score": float, // Score for sector spread
    "asset_class_diversification_score": float, // Score for asset class spread
    "geographic_diversification_score": float, // Score for geographic spread
    "ml_diversification_insights": {}, // ML-driven insights for optimal diversification
    "concentration_risk": float,       // Risk due to over-concentration in specific assets/sectors
    "diversification_score": int,      // Overall diversification score (0-100)
    "top_concentrated_sectors": ["str"], // List of most concentrated sectors
    "top_concentrated_assets": ["str"]   // List of most concentrated assets
  },
  "performance": {
    "total_return": float,             // Overall return since inception/period start
    "annualized_return": float,        // Annualized return
    "ytd_return": float,               // Year-to-date return
    "monthly_returns": [float],        // List of historical monthly returns
    "benchmark_comparison": float,     // Performance relative to a benchmark
    "alpha": float,                    // Excess return over benchmark
    "tracking_error": float,           // Volatility of excess returns
    "performance_score": int,          // Overall performance score (0-100)
    "ml_performance_prediction": float, // ML-predicted future performance
    "risk_adjusted_returns": float,    // Returns adjusted for risk
    "performance_consistency": float,  // Consistency of returns over time
    "market_timing_score": float       // Score for effective market timing
  },
  "recommendations": [
    {
      "type": "BUY" | "SELL" | "HOLD" | "REBALANCE", // Type of recommendation
      "asset_symbol": "str",           // Target asset symbol
      "reason": "str",                 // Explanation for the recommendation
      "impact_score": float,           // Predicted impact of following the recommendation
      "target_allocation": float       // For rebalance recommendations, target percentage
    }
  ],
  "overall_score": int,                // Aggregate score for the portfolio (0-100)
  "summary": "str",                    // A concise textual summary of the analysis
  "analysis_timestamp": "str",         // ISO 8601 formatted timestamp of when the analysis was performed
  "ml_analysis_available": bool,       // Indicates if full ML analysis was successfully performed
  "image_analysis": {},                // Detailed results from the deep learning image processing
  "advanced_insights": {}              // Additional deep learning-generated insights
}
```

#### Error Responses:

*   `400 Bad Request`: Indicates invalid input, such as malformed holdings data, an unsupported image format, or missing required parameters.
*   `500 Internal Server Error`: An unexpected error occurred during the server-side processing.
*   `503 Service Unavailable`: The service is temporarily unable to handle the request, possibly due to issues with dependent external services (e.g., market data APIs, ML model servers).

## ⚙️ Configuration

The AI-Powered Portfolio Analyzer can be configured using environment variables, allowing for flexible deployment and management.

*   **`LOG_LEVEL`**: (Default: `INFO`) Controls the verbosity of logging output. Accepted values include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
*   **`MARKET_DATA_API_KEY`**: (Required for real-time data) Your API key for accessing external real-time market data providers. This is crucial for accurate and up-to-date analysis.
*   **`ML_MODEL_PATH`**: (Default: `./models/`) Specifies the file system path where pre-trained machine learning models (for image recognition, risk prediction, etc.) are stored.
*   **`IMAGE_PROCESSING_CONFIG`**: (JSON string, optional) A JSON string containing specific configuration parameters for the image analysis module, such as OCR settings, object detection thresholds, or pre-processing steps.
*   **`FALLBACK_ENABLED`**: (Default: `True`) A boolean flag to enable or disable the automatic fallback to basic analysis if the enhanced AI analysis encounters unrecoverable errors.

Example `.env` file for local development:

```dotenv
LOG_LEVEL=DEBUG
MARKET_DATA_API_KEY=YOUR_SECURE_MARKET_DATA_API_KEY_HERE
ML_MODEL_PATH=./ml_models/
FALLBACK_ENABLED=True
IMAGE_PROCESSING_CONFIG='{"ocr_engine": "tesseract", "min_confidence": 0.7}'
```

## 🤝 Contributing

We welcome and appreciate contributions to the AI-Powered Portfolio Analyzer! If you're interested in improving this project, please follow these guidelines:

1.  **Fork the repository** to your GitHub account.
2.  **Clone your forked repository** to your local machine:
    ```bash
    git clone https://github.com/YourUsername/ai-portfolio-analyzer.git
    cd ai-portfolio-analyzer
    ```
3.  **Create a new branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-awesome-feature
    ```
4.  **Make your changes.** Ensure your code adheres to good Python practices (e.g., PEP 8) and is well-documented. Consider using a linter like `flake8` and a formatter like `black`.
5.  **Write comprehensive unit and integration tests** for your changes to ensure functionality and prevent regressions.
6.  **Commit your changes** with a clear, concise, and descriptive commit message:
    ```bash
    git commit -m "feat: Add new deep learning model for enhanced image analysis"
    ```
7.  **Push your branch** to your forked repository:
    ```bash
    git push origin feature/your-awesome-feature
    ```
8.  **Open a Pull Request** against the `main` branch of the original repository. Provide