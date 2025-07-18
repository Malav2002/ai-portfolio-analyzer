// utils/formatters.ts
// Create this file in your frontend/src/utils/ directory
// This provides centralized, safe formatting functions for the entire app

/**
 * Safe number formatting utilities that prevent crashes from undefined/null values
 */

export const safeFormatNumber = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0';
  }
  return num.toLocaleString();
};

export const safeFormatCurrency = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '$0.00';
  }
  return `$${num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
};

export const safeFormatPercentage = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0.00%';
  }
  return `${num.toFixed(2)}%`;
};

export const safeFormatDecimal = (num: any, decimals: number = 2): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0.00';
  }
  return num.toFixed(decimals);
};

export const safeFormatFileSize = (bytes: any): string => {
  if (typeof bytes !== 'number' || isNaN(bytes) || bytes === null || bytes === undefined) {
    return '0 KB';
  }
  
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  if (bytes === 0) return '0 KB';
  
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const formattedSize = (bytes / Math.pow(1024, i)).toFixed(2);
  
  return `${formattedSize} ${sizes[i]}`;
};

export const safeFormatLargeNumber = (num: any): string => {
  if (typeof num !== 'number' || isNaN(num) || num === null || num === undefined) {
    return '0';
  }
  
  if (num >= 1e9) {
    return `${(num / 1e9).toFixed(1)}B`;
  } else if (num >= 1e6) {
    return `${(num / 1e6).toFixed(1)}M`;
  } else if (num >= 1e3) {
    return `${(num / 1e3).toFixed(1)}K`;
  }
  
  return num.toString();
};

// Safe mathematical operations
export const safeAdd = (a: any, b: any): number => {
  const numA = typeof a === 'number' && !isNaN(a) ? a : 0;
  const numB = typeof b === 'number' && !isNaN(b) ? b : 0;
  return numA + numB;
};

export const safeSubtract = (a: any, b: any): number => {
  const numA = typeof a === 'number' && !isNaN(a) ? a : 0;
  const numB = typeof b === 'number' && !isNaN(b) ? b : 0;
  return numA - numB;
};

export const safeDivide = (a: any, b: any): number => {
  const numA = typeof a === 'number' && !isNaN(a) ? a : 0;
  const numB = typeof b === 'number' && !isNaN(b) && b !== 0 ? b : 1;
  return numA / numB;
};

export const safeMultiply = (a: any, b: any): number => {
  const numA = typeof a === 'number' && !isNaN(a) ? a : 0;
  const numB = typeof b === 'number' && !isNaN(b) ? b : 0;
  return numA * numB;
};

// Safe percentage calculations
export const safeCalculatePercentage = (value: any, total: any): number => {
  const numValue = typeof value === 'number' && !isNaN(value) ? value : 0;
  const numTotal = typeof total === 'number' && !isNaN(total) && total !== 0 ? total : 1;
  return (numValue / numTotal) * 100;
};

export const safeCalculateChange = (current: any, previous: any): number => {
  const numCurrent = typeof current === 'number' && !isNaN(current) ? current : 0;
  const numPrevious = typeof previous === 'number' && !isNaN(previous) && previous !== 0 ? previous : 1;
  return ((numCurrent - numPrevious) / numPrevious) * 100;
};

// Data validation helpers
export const isValidNumber = (value: any): boolean => {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
};

export const isValidPositiveNumber = (value: any): boolean => {
  return isValidNumber(value) && value >= 0;
};

export const isValidPercentage = (value: any): boolean => {
  return isValidNumber(value) && value >= 0 && value <= 100;
};

// Safe array operations
export const safeSum = (arr: any[]): number => {
  if (!Array.isArray(arr)) return 0;
  return arr.reduce((sum, item) => safeAdd(sum, item), 0);
};

export const safeMax = (arr: any[]): number => {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  const validNumbers = arr.filter(isValidNumber);
  return validNumbers.length > 0 ? Math.max(...validNumbers) : 0;
};

export const safeMin = (arr: any[]): number => {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  const validNumbers = arr.filter(isValidNumber);
  return validNumbers.length > 0 ? Math.min(...validNumbers) : 0;
};

export const safeAverage = (arr: any[]): number => {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  const validNumbers = arr.filter(isValidNumber);
  return validNumbers.length > 0 ? safeSum(validNumbers) / validNumbers.length : 0;
};

// Export all functions as default for easy importing
export default {
  safeFormatNumber,
  safeFormatCurrency,
  safeFormatPercentage,
  safeFormatDecimal,
  safeFormatFileSize,
  safeFormatLargeNumber,
  safeAdd,
  safeSubtract,
  safeDivide,
  safeMultiply,
  safeCalculatePercentage,
  safeCalculateChange,
  isValidNumber,
  isValidPositiveNumber,
  isValidPercentage,
  safeSum,
  safeMax,
  safeMin,
  safeAverage
};