This document provides comprehensive API documentation for the AI Portfolio Analyzer ML Service.

---

# AI Portfolio Analyzer ML Service API Documentation

## 1. Overview and Purpose

The AI Portfolio Analyzer ML Service is the core intelligence layer of an AI-powered application designed to analyze investment portfolios. Its primary purpose is to process portfolio data, enrich it with real-time market information, and generate AI-driven insights and predictions. This service handles complex tasks such as optical character recognition (OCR) for parsing portfolio screenshots, fetching up-to-date financial data from various sources, and applying machine learning models for portfolio assessment.

This API is primarily built using Python with FastAPI, providing asynchronous capabilities and robust error handling.

**Key Features:**
*   **Portfolio Data Parsing**: Capable of processing raw input (e.g., screenshot images) to extract structured portfolio holdings.
*   **Real-time Market Data Enrichment**: Integrates with multiple external financial data providers (YFinance, Alpha Vantage, FMP, Polygon) to fetch current prices, market caps, sector information, and more for individual stock holdings. Includes rate-limiting and retry mechanisms for reliable data acquisition.
*   **AI-Powered Analysis**: Generates comprehensive portfolio analysis, including overall scores, diversification scores, risk assessments, and performance outlooks.
*   **Debug Capabilities**: Provides a dedicated endpoint for detailed logging and troubleshooting of the entire analysis pipeline, invaluable for development and issue resolution.

**Base URL**: (Assumed, typically `http://localhost:8000` during development, or a deployed cloud endpoint)

## 2. Functions/Classes

This section details the exposed API endpoints and key internal service functions that define the capabilities of the ML service.

### API Endpoints

#### `GET /`

*   **Description**: The root endpoint of the ML service. It provides a basic health check and general information about the service, including its version, status, and a list of key available endpoints.
*   **Signature**: `GET /`
*   **Parameters**: None
*   **Return Value**: `Dict` - A JSON object containing service metadata.

#### `POST /api/debug/analyze-portfolio`

*   **Description**: A specialized debug endpoint designed for extensive logging and troubleshooting of the portfolio analysis process. It accepts an uploaded portfolio screenshot image file and returns a highly detailed report. This report includes information on file processing, OCR results, portfolio data parsing, market data enrichment, and the final AI analysis, along with any errors or warnings encountered at each stage. This endpoint is crucial for developers to understand the pipeline's behavior and diagnose issues.
*   **Signature**: `POST /api/debug/analyze-portfolio(file: UploadFile)`
*   **Parameters**:
    *   `file`: `UploadFile` - The portfolio screenshot image file to be analyzed.
*   **Return Value**: `Dict` - A comprehensive JSON debug report.

### Core Service Functions (Internal to ML Service)

These functions are not directly exposed as API endpoints but are fundamental components that define the capabilities and internal workings of the ML service. They are called by the API endpoints to perform the core logic.

#### `generate_basic_ai_analysis`

*   **Description**: A synchronous helper function that generates a simplified, fallback AI analysis for a given portfolio. This function is typically used when full market data enrichment or complex AI models are unavailable, or when a quick, high-level assessment is sufficient. It calculates basic scores based on the number of holdings and total value.
*   **Signature**: `generate_basic_ai_analysis(portfolio_data: Dict) -> Dict`
*   **Parameters**:
    *   `portfolio_data`: `Dict` - A dictionary containing parsed portfolio information.
*   **Return Value**: `Dict` - A dictionary containing basic AI analysis scores and insights.

#### `MarketDataService`

*   **Description**: This class provides an enhanced service for fetching and enriching financial market data. It is designed to be robust, incorporating internal rate-limiting, retry mechanisms, and fallback options to handle external API constraints, network issues, and data unavailability. It aggregates data from multiple sources to provide comprehensive market information for portfolio holdings.
*   **Signature**: `class MarketDataService:`
*   **Methods**:
    *   `__init__(self)`:
        *   **Description**: The constructor for the `MarketDataService`. It initializes API keys for various financial data providers (Alpha Vantage, FMP, Polygon), sets up rate-limiting parameters (minimum delay between requests, maximum retries, retry delay), and prepares an `aiohttp` session for asynchronous HTTP requests.
        *   **Parameters**: None
        *   **Return Value**: None
    *   `enrich_portfolio_data(self, portfolio_data: Dict) -> Dict`:
        *   **Description**: An asynchronous method that takes a portfolio's structured data and enriches each holding with real-time market information. It iterates through the holdings, fetches data for each symbol using a prioritized list of data sources, and applies rate-limiting to prevent exceeding API quotas. If market data is unavailable after retries, it can provide a basic fallback quote.
        *   **Signature**: `async def enrich_portfolio_data(self, portfolio_data: Dict) -> Dict`
        *   **Parameters**:
            *   `portfolio_data`: `Dict` - A dictionary representing the portfolio. It is expected to contain a `'holdings'` key, which is a list of dictionaries, where each holding dictionary must include a `'symbol'` field (e.g., `'AAPL'`, `'GOOG'`).
        *   **Return Value**: `Dict` - The original `portfolio_data` dictionary, but with the `holdings` list updated to include `enriched