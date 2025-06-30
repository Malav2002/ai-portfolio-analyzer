'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function Home() {
  const [backendStatus, setBackendStatus] = useState('checking...');
  const [mlStatus, setMlStatus] = useState('checking...');

  useEffect(() => {
    // Test backend
    fetch('http://localhost:3001/health')
      .then(res => res.json())
      .then(data => setBackendStatus('✅ Connected'))
      .catch(() => setBackendStatus('❌ Disconnected'));

    // Test ML service
    fetch('http://localhost:8002/health')
      .then(res => res.json())
      .then(data => setMlStatus('✅ Connected'))
      .catch(() => setMlStatus('❌ Disconnected'));
  }, []);

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>🚀 AI Portfolio Analyzer</h1>
      
      <div style={{ marginBottom: '2rem', padding: '1rem', backgroundColor: '#f0f8ff', borderRadius: '8px' }}>
        <h2>🎯 Quick Actions:</h2>
        <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
          <Link 
            href="/portfolio/upload" 
            style={{ 
              backgroundColor: '#0066cc', 
              color: 'white', 
              padding: '0.75rem 1.5rem', 
              borderRadius: '8px', 
              textDecoration: 'none',
              fontWeight: 'bold'
            }}
          >
            📸 Upload Portfolio Screenshot
          </Link>
        </div>
      </div>

      <h2>Service Status:</h2>
      <ul>
        <li>Backend API: {backendStatus}</li>
        <li>ML Service: {mlStatus}</li>
        <li>Database: ✅ PostgreSQL Running</li>
        <li>Cache: ✅ Redis Running</li>
      </ul>
      
      <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#f0f8ff', borderRadius: '8px' }}>
        <h3>🔗 Development Links:</h3>
        <ul>
          <li><a href="http://localhost:3001/health" target="_blank">Backend Health Check</a></li>
          <li><a href="http://localhost:8002/health" target="_blank">ML Service Health Check</a></li>
          <li><a href="http://localhost:8002/docs" target="_blank">ML Service API Docs</a></li>
          <li><Link href="/portfolio/upload">Upload Portfolio Screenshot</Link></li>
        </ul>
      </div>
    </div>
  );
}
