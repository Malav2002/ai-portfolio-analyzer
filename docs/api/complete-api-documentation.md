This document provides comprehensive API documentation for the AI Portfolio Analyzer, an application designed to analyze portfolio screenshots using AI, integrate real-time market data, and offer predictive insights.

---

## AI Portfolio Analyzer API Documentation

### 1. Overview and Purpose

The AI Portfolio Analyzer is an advanced application that leverages Artificial Intelligence, Optical Character Recognition (OCR), and real-time market data to provide in-depth analysis of investment portfolios from screenshots. Users can upload an image of their portfolio, and the system will automatically extract holdings, fetch current market prices, calculate performance metrics, and generate AI-driven insights and recommendations.

The API serves as the interface to the core machine learning and data processing capabilities of the system. It enables:
*   **Automated Data Extraction:** Using OCR to read portfolio details from images.
*   **Portfolio Parsing:** Structuring the extracted text into actionable financial data.
*   **Market Data Enrichment:** Integrating real-time stock quotes and financial metrics.
*   **AI-Powered Analysis:** Generating scores, insights, and recommendations based on the parsed and enriched data.
*   **Debugging and Troubleshooting:** Dedicated endpoints for inspecting intermediate steps and raw data.

The system is composed of a frontend (React/Next.js), a backend (Node.js/Express), and an ML service (Python/FastAPI). This documentation primarily focuses on the ML service's API endpoints, which are consumed by the backend and can also be accessed directly for development or integration purposes.

### 2. Functions/Classes (API Endpoints)

The following API endpoints are exposed by the ML service:

#### 2.1. `GET /` - Root Endpoint

*   **Description:** Provides basic information about the ML service, its version, operational status, and the availability of its core sub-services (like OCR). It also lists other key endpoints.
*   **HTTP Method:** `GET`
*   **URL:** `/`

#### 2.2. `POST /api/ocr/parse-portfolio-with-market-data` - Enhanced Portfolio Analysis

*   **Description:** This is the primary endpoint for comprehensive portfolio analysis. It takes an image file of a portfolio screenshot, performs OCR to extract text, parses the text into structured portfolio data, enriches this data with real-time market information, and finally generates an AI-driven analysis with insights and recommendations. It includes robust error handling and logging.
*   **HTTP Method:** `POST`
*   **URL:** `/api/ocr/parse-portfolio-with-market-data`

#### 2.3. `POST /api/debug/analyze-portfolio` - Debug Portfolio Analysis

*   **Description:** A specialized debug endpoint that performs the full portfolio analysis pipeline (OCR, parsing, market data, AI analysis) but returns an extensive `debug_report`. This report includes detailed logs, intermediate results, service statuses, and raw outputs at each stage, making it invaluable for troubleshooting and understanding the analysis process.
*   **HTTP Method:** `POST`
*   **URL:** `/api/debug/analyze-portfolio`

#### 2.4. `POST /api/debug/ocr-only` - Debug OCR Extraction

*   **Description:** A debug endpoint focused solely on the OCR (Optical Character Recognition) process. It takes an image file and returns the raw text extracted by the OCR service, along with the OCR result object. This endpoint bypasses portfolio parsing, market data enrichment, and AI analysis, allowing developers to inspect the initial text extraction step.
*   **HTTP Method:** `POST`
*   **URL:** `/api/debug/ocr-only`

### 3. Parameters

#### 3.1. `GET /`

*   **No parameters.**

#### 3.2. `POST /api/ocr/parse-portfolio-with-market-data`

*   **Parameter:** `file`
    *   **Type:** `UploadFile` (FastAPI/Python) / `File` (HTTP multipart/form-data)
    *   **Description:** The image file (e.g., PNG, JPEG) of the portfolio screenshot to be analyzed.
    *   **Required:** Yes

#### 3.3. `POST /api/debug/analyze-portfolio`

*   **Parameter:** `file`
    *   **Type:** `UploadFile` (FastAPI/Python) / `File` (HTTP multipart/form-data)
    *   **Description:** The image file (e.g., PNG, JPEG) of the portfolio screenshot for which a debug analysis is requested.
    *   **Required:** Yes

#### 3.4. `POST /api/debug/ocr-only`

*   **Parameter:** `file`
    *   **Type:** `UploadFile` (FastAPI/Python) / `File` (HTTP multipart/form-data)
    *   **Description:** The image file (e.g., PNG, JPEG) from which text needs to be extracted via OCR.
    *   **Required:** Yes

### 4. Return Values

#### 4.1. `GET /`

*   **Type:** `application/json`
*   **Description:** A JSON object containing metadata about the ML service.
*   **Example Structure:**
    ```json
    {
        "message": "AI Portfolio Analyzer ML Service",
        "version": "2.0.0",
        "status": "running",
        "services_available": true,
        "ocr_available": true,
        "directory_structure": "src/services/",
        "endpoints": {
            "health": "/health",
            "portfolio_analysis": "/api/ocr/parse-portfolio-with-market-data",
            "portfolio_analyze": "/api/portfolio/analyze"
        }
    }
    ```

#### 4.2. `POST /api/ocr/parse-portfolio-with-market-data`

*   **Type:** `application/json`
*   **Description:** A comprehensive JSON object detailing the success of the analysis, the parsed portfolio data, market data enrichment status, AI analysis results, and any errors or messages. The backend's `normalizePortfolioResponse` function ensures a consistent structure.
*   **Example Structure:**
    ```json
    {
        "success": true,
        "analysis": {
            "portfolio_data": {
                "holdings_count": 3,
                "total_value": 15000.00,
                "holdings": [
                    {
                        "symbol": "AAPL",
                        "name": "Apple Inc.",
                        "shares": 10,
                        "average_cost": 150.00,
                        "current_price": 175.00,
                        "market_value": 1750.00,
                        "daily_change": 5.25,
                        "daily_change_percent": 3.09,
                        "total_gain_loss": 250.00,
                        "total_gain_loss_percent": 16.67,
                        "currency": "USD",
                        "last_updated": "2023-10-27T10:30:00Z"
                    },
                    {
                        "symbol": "MSFT",
                        "name": "Microsoft Corp.",
                        "shares": 5,
                        "average_cost": 300.00,
                        "current_price": 320.00,
                        "market_value": 1600.00,
                        "daily_change": 2.00,
                        "daily_change_percent": 0.63,
                        "total_gain_loss": 100.00,
                        "total_gain_loss_percent": 6.67,
                        "currency": "USD",
                        "last_updated": "2023-10-27T10:30:00Z"
                    },
                    {
                        "symbol": "GOOGL",
                        "name": "Alphabet Inc. (Class A)",
                        "shares": 2,
                        "average_cost": 120.00,
                        "current_price": 130.00,
                        "market_value": 260.00,
                        "daily_change": 1.00,
                        "daily_change_percent": 0.77,
                        "total_gain_loss": 20.00,
                        "total_gain_loss_percent": 8.33,
                        "currency": "USD",
                        "last_updated": "2023-10-27T10:30:00Z"
                    }
                ],
                "summary": {
                    "total_daily_change": 8.25,
                    "total_daily_change_percent": 1.5,
                    "total_gain_loss": 370.00,
                    "total_gain_loss_percent": 10.5
                }
            },
            "ai_analysis": {
                "overall_score": 8.5,
                "diversification_score": 7.8,
                "risk_score": 6.2,
                "performance_outlook": "Positive with moderate growth potential",
                "key_insights": [
                    "Portfolio shows strong performance in tech sector.",
                    "Diversification could be improved by adding exposure to other industries.",
                    "Total value: $15,000.00"
                ],
                "recommendations": [
                    "Consider adding holdings in healthcare or consumer staples for better balance.",
                    "Review risk tolerance and adjust allocations accordingly."
                ]
            }
        },
        "market_data_status": {
            "success": true,
            "message": "Market data fetched successfully for all holdings."
        },
        "ocr_status": {
            "success": true,
            "message": "OCR extraction completed successfully."
        },
        "message": "Portfolio analysis completed successfully."
    }
    ```
    *Note: The `generate_basic_ai_analysis` function (an internal ML service function) provides a fallback structure for `ai_analysis` if the primary AI model is unavailable or fails.*

#### 4.3. `POST /api/debug/analyze-portfolio`

*   **Type:** `application/json`
*   **Description:** A detailed debug report containing timestamps, file information, service statuses, raw OCR output, parsed portfolio data before market enrichment, market data fetch results, and the final AI analysis.
*   **Example Structure:**
    ```json
    {
        "timestamp": "2023-10-27T10:30:00.123456Z",
        "file_info": {
            "filename": "portfolio_screenshot.png",
            "content_type": "image/png",
            "size_bytes": 1234567
        },
        "service_status": {
            "ocr_service_available": true,
            "portfolio_parser_available": true,
            "market_data_service_available": true,
            "ai_analyzer_available": true,
            "db_service_available": false
        },
        "ocr_result": {
            "success": true,
            "extracted_text": "AAPL 10 150.00 ... MSFT 5 300.00 ...",
            "raw_ocr_output": {
                // Raw output from the OCR engine (e.g., Google Vision API response)
                "text_annotations": [
                    { "description": "AAPL", "bounding_poly": { "vertices": [...] } },
                    // ... more annotations
                ]
            }
        },
        "parsed_portfolio_data_raw": {
            "success": true,
            "holdings": [
                { "symbol": "AAPL", "shares": 10, "cost_basis": 150.00 },
                // ...
            ],
            "total_value_ocr": 15000.00,
            "confidence_score": 0.95
        },
        "market_data_enrichment": {
            "success": true,
            "message": "Market data fetched for 3 out of 3 holdings.",
            "enriched_holdings": [
                {
                    "symbol": "AAPL",
                    "shares": 10,
                    "average_cost": 150.00,
                    "current_price": 175.00,
                    "market_value": 1750.00,
                    "daily_change": 5.25,
                    "daily_change_percent": 3.09
                }
                // ... other holdings with market data
            ],
            "failed_symbols": []
        },
        "ai_analysis_result": {
            "success": true,
            "analysis": {
                "overall_score": 8.5,
                "diversification_score": 7.8,
                "risk_score": 6.2,
                "performance_outlook": "Positive with moderate growth potential",
                "key_insights": [
                    "Portfolio shows strong performance in tech sector.",
                    "Diversification could be improved by adding exposure to other industries."
                ],
                "recommendations": [
                    "Consider adding holdings in healthcare or consumer staples for better balance."
                ]
            }
        },
        "final_response_structure": {
            // This section would mirror the structure of the /api/ocr/parse-portfolio-with-market-data endpoint
            // but is included here for completeness of the debug report.
            "success": true,
            "analysis": { /* ... */ },
            "market_data_status": { /* ... */ },
            "ocr_status": { /* ... */ },
            "message": "Debug analysis completed successfully."
        },
        "logs": [
            "INFO: Processing portfolio image: portfolio_screenshot.png",
            "INFO: File size: 1234567 bytes",
            "DEBUG: OCR service available: True",
            // ... extensive log entries
        ]
    }
    ```

#### 4.4. `POST /api/debug/ocr-only`

*   **Type:** `application/json`
*   **Description:** A JSON object containing the success status, the raw OCR result object, and the concatenated extracted text.
*   **Example Structure:**
    ```json
    {
        "success": true,
        "ocr_result": {
            // Raw output from the OCR engine (e.g., Google Vision API response)
            "text_annotations": [
                { "description": "My Portfolio", "bounding_poly": { "vertices": [...] } },
                { "description": "AAPL", "bounding_poly": { "vertices": [...] } },
                { "description": "10", "bounding_poly": { "vertices": [...] } },
                // ... more detailed OCR results
            ],
            "full_text_annotation": {
                "text": "My Portfolio\nAAPL 10 shares\nMSFT 5 shares\n..."
            }
        },
        "extracted_text": "My Portfolio\nAAPL 10 shares\nMSFT 5 shares\nTotal Value: $15,000.00",
        "message": "OCR extraction completed successfully."
    }
    ```

### 5. Usage Examples

#### 5.1. `GET /` - Root Endpoint (ML Service Health Check)

**Using `curl`:**
```bash
curl http://localhost:8000/
```

**Expected Response:**
```json
{
    "message": "AI Portfolio Analyzer ML Service",
    "version": "2.0.0",
    "status": "running",
    "services_available": true,
    "ocr_available": true,
    "directory_structure": "src/services/",
    "endpoints": {
        "health": "/health",
        "portfolio_analysis": "/api/ocr/parse-portfolio-with-market-data",
        "portfolio_analyze": "/api/portfolio/analyze"
    }
}
```

#### 5.2. `POST /api/ocr/parse-portfolio-with-market-data` - Enhanced Portfolio Analysis

**Using JavaScript (Frontend/Backend):**
This example demonstrates how the frontend (or backend) would send a portfolio screenshot to the ML service.

```javascript
// In backend/routes/ai-portfolio.js or frontend component
import axios from 'axios';
import FormData from 'form-data'; // For Node.js backend, browser has native FormData

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000'; // Example ML service URL

async function analyzePortfolioScreenshot(imageFile) {
  const formData = new FormData();
  formData.append('file', imageFile, imageFile.name); // imageFile should be a File object (browser) or Buffer (Node.js)

  try {
    console.log(`Sending image to ML service at ${ML_SERVICE_URL}/api/ocr/parse-portfolio-with-market-data`);
    const response = await axios.post(
      `${ML_SERVICE_URL}/api/ocr/parse-portfolio-with-market-data`,
      formData,
      {
        headers: {
          ...formData.getHeaders(), // Important for Node.js FormData
          'Content-Type': 'multipart/form-data' // Explicitly set for clarity
        },
        maxBodyLength: Infinity, // Allow large files
        maxContentLength: Infinity
      }
    );
    console.log('ML Service Response:', response.data);
    
    // The backend would then normalize this response using normalizePortfolioResponse
    // For example:
    // const normalizedData = normalizePortfolioResponse(response);
    // return normalizedData;

    return response.data;

  } catch (error) {
    console.error('Error during portfolio analysis:', error.response ? error.response.data : error.message);
    throw error;
  }
}

// Example usage (assuming imageFile is available, e.g., from an <input type="file"> event)
// const fileInput = document.getElementById('portfolioImageInput');
// fileInput.addEventListener('change', async (event) => {
//   const selectedFile = event.target.files[0];
//   if (selectedFile) {
//     try {
//       const analysis = await analyzePortfolioScreenshot(selectedFile);
//       console.log('Final Analysis:', analysis);
//       // Update UI with analysis results
//     } catch (err) {
//       console.error('Failed to analyze portfolio:', err);
//       // Show error to user
//     }
//   }
// });
```

**Using `curl` (Direct to ML Service):**
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/portfolio_screenshot.png" \
  http://localhost:8000/api/ocr/parse-portfolio-with-market-data
```

#### 5.3. `POST /api/debug/analyze-portfolio` - Debug Portfolio Analysis

**Using `curl`:**
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/portfolio_screenshot.png" \
  http://localhost:8000/api/debug/analyze-portfolio
```

**Expected Response (truncated for brevity, see full structure above):**
```json
{
    "timestamp": "2023-10-27T10:30:00.123456Z",
    "file_info": {
        "filename": "portfolio_screenshot.png",
        "content_type": "image/png",
        "size_bytes": 1234567
    },
    "service_status": {
        "ocr_service_available": true,
        "portfolio_parser_available": true,
        "market_data_service_available": true,
        "ai_analyzer_available": true,
        "db_service_available": false
    },
    "ocr_result": { /* ... */ },
    "parsed_portfolio_data_raw": { /* ... */ },
    "market_data_enrichment": { /* ... */ },
    "ai_analysis_result": { /* ... */ },
    "final_response_structure": { /* ... */ },
    "logs": [
        "INFO: Processing portfolio image: portfolio_screenshot.png",
        "INFO: File size: 1234567 bytes",
        // ... extensive log entries
    ]
}
```

#### 5.4. `POST /api/debug/ocr-only` - Debug OCR Extraction

**Using `curl`:**
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/portfolio_screenshot.png" \
  http://localhost:8000/api/debug/ocr-only
```

**Expected Response:**
```json
{
    "success": true,
    "ocr_result": {
        "text_annotations": [
            { "description": "My Portfolio", "bounding_poly": { "vertices": [...] } },
            { "description": "AAPL", "bounding_poly": { "vertices": [...] } }
        ],
        "full_text_annotation": {
            "text": "My Portfolio\nAAPL 10 shares\nMSFT 5 shares\nTotal Value: $15,000.00"
        }
    },
    "extracted_text": "My Portfolio\nAAPL 10 shares\nMSFT 5 shares\nTotal Value: $15,000.00",
    "message": "OCR extraction completed successfully."
}
```

### 6. Error Handling

The ML service utilizes FastAPI's `HTTPException` for structured error responses and general `Exception` handling for internal issues. The responses typically include a `success: false` flag, an `error` message, and sometimes a more detailed `message` field.

#### 6.1. Common HTTP Status Codes

*   **`200 OK`**: Request successful.
*   **`400 Bad Request`**: Invalid input, e.g., no file provided, unsupported file type (though the API expects an image).
    *   **Example:**
        ```json
        {
            "success": false,
            "error": "Validation Error",
            "message": "No 'file' part in the request."
        }
        ```
*   **`404 Not Found`**: Endpoint does not exist.
*   **`422 Unprocessable Entity`**: FastAPI's validation error for incorrect data types or missing required fields.
    *   **Example:**
        ```json
        {
            "detail": [
                {
                    "loc": [ "body", "file" ],
                    "msg": "field required",
                    "type": "value_error.missing"
                }
            ]
        }
        ```
*   **`500 Internal Server Error`**: An unexpected error occurred on the server. This could be due to:
    *   **Service Unavailability:** If a critical internal service (OCR, Market Data, AI Analyzer) is not initialized or fails.
        *   **Example (OCR service unavailable):**
            ```json
            {
                "success": false,
                "error": "Service Unavailable",
                "message": "OCR service is not available. Please check ML service health."
            }
            ```
        *   **Example (Market data service error):**
            ```json
            {
                "success": false,
                "error": "Market Data Error",
                "message": "Failed to fetch market data for some symbols: ['XYZ']. Please try again later."
            }
            ```
    *   **Processing Failure:** Issues during image processing, OCR, portfolio parsing, or AI analysis.
        *   **Example (Parsing error):**
            ```json
            {
                "success": false,
                "error": "Portfolio Parsing Error",
                "message": "Could not parse portfolio data from the extracted text. The image might be unclear or in an unsupported format."
            }
            ```
    *   **General Uncaught Exception:**
        ```json
        {
            "success": false,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during processing. Please contact support."
        }
        ```

#### 6.2. Frontend/Backend Error Handling

The frontend component `AnalysisResults.tsx` specifically checks for `data?.success` and displays an error message if `false`, utilizing `data?.error` or `data?.message`.

```typescript
// In frontend/src/app/components/AnalysisResults.tsx
if (!data?.success) {
  return (
    <div className="glass-effect rounded-3xl p-8 border border-red-200">
      <div className="flex items-center space-x-4">
        <div className="p-3 bg-gradient-to-br from-red-500 to-rose-600 rounded-2xl">
          <AlertTriangle className="w-8 h-8 text-white" />
        </div>
        <div>
          <h4 className="font-bold text-red-800 text-lg">Analysis Failed</h4>
          <p className="text-red-700">{data?.error || data?.message || "An unknown error occurred during analysis."}</p>
        </div>
      </div>
    </div>
  );
}
```

### 7. Common Use Cases

#### 7.1. Analyzing a New Portfolio Screenshot

*   **