const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'portfolio-analyzer-backend'
  });
});

// Test database connection
app.get('/api/test-db', async (req, res) => {
  try {
    // TODO: Add actual database connection test
    res.json({ message: 'Database connection will be tested here' });
  } catch (error) {
    res.status(500).json({ error: 'Database connection failed' });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Backend server running on port ${PORT}`);
});
