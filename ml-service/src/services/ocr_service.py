import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import logging
from typing import Dict, List, Tuple, Optional
import io

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        # Different Tesseract configs for different layouts
        self.table_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        self.general_config = r'--oem 3 --psm 6'
        
    def preprocess_for_table(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image specifically for table data
        """
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert PIL to OpenCV format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply different preprocessing for better table recognition
            
            # 1. Increase contrast more aggressively
            alpha = 2.0  # Contrast control (higher than before)
            beta = -50   # Brightness control 
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            
            # 2. Apply morphological operations to preserve table structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # 3. Use different thresholding
            # Try Otsu's thresholding which works well for documents
            _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh
            
        except Exception as e:
            logger.error(f"Table preprocessing failed: {e}")
            # Return original image as fallback
            pil_image = Image.open(io.BytesIO(image_bytes))
            return np.array(pil_image.convert('L'))

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Standard preprocessing for general text
        """
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert PIL to OpenCV format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Increase contrast
            alpha = 1.5  # Contrast control
            beta = 0     # Brightness control
            enhanced = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original image as fallback
            pil_image = Image.open(io.BytesIO(image_bytes))
            return np.array(pil_image.convert('L'))

    async def extract_text_with_layout(self, image_bytes: bytes) -> Dict:
        """
        Extract text preserving layout structure (better for tables)
        """
        try:
            # Try table-optimized preprocessing
            processed_image = self.preprocess_for_table(image_bytes)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Use TSV output to preserve positioning
            tsv_data = pytesseract.image_to_data(
                pil_image, 
                config=self.table_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Reconstruct text with better spacing
            words_by_line = {}
            min_confidence = 30
            
            for i in range(len(tsv_data['text'])):
                if int(tsv_data['conf'][i]) > min_confidence:
                    word = tsv_data['text'][i].strip()
                    if word:
                        line_num = tsv_data['line_num'][i]
                        if line_num not in words_by_line:
                            words_by_line[line_num] = []
                        
                        words_by_line[line_num].append({
                            'text': word,
                            'left': tsv_data['left'][i],
                            'confidence': int(tsv_data['conf'][i])
                        })
            
            # Sort words within each line by position
            reconstructed_lines = []
            total_confidence = 0
            total_words = 0
            
            for line_num in sorted(words_by_line.keys()):
                words = sorted(words_by_line[line_num], key=lambda x: x['left'])
                line_text = ' '.join([w['text'] for w in words])
                reconstructed_lines.append(line_text)
                
                for word in words:
                    total_confidence += word['confidence']
                    total_words += 1
            
            full_text = '\n'.join(reconstructed_lines)
            avg_confidence = total_confidence / total_words if total_words > 0 else 0
            
            return {
                'method': 'tesseract_layout',
                'text': full_text,
                'confidence': avg_confidence / 100,
                'word_count': total_words,
                'lines': reconstructed_lines
            }
            
        except Exception as e:
            logger.error(f"Layout-aware OCR failed: {e}")
            return await self.extract_text_tesseract(image_bytes)

    async def extract_text_tesseract(self, image_bytes: bytes) -> Dict:
        """
        Standard Tesseract extraction (fallback)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_bytes)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.general_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter out low confidence text
            min_confidence = 30
            extracted_text = []
            total_confidence = 0
            valid_words = 0
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > min_confidence:
                    word = data['text'][i].strip()
                    if word:
                        extracted_text.append({
                            'text': word,
                            'confidence': int(data['conf'][i]),
                            'bbox': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        })
                        total_confidence += int(data['conf'][i])
                        valid_words += 1
            
            # Calculate average confidence
            avg_confidence = total_confidence / valid_words if valid_words > 0 else 0
            
            # Combine all text
            full_text = ' '.join([item['text'] for item in extracted_text])
            
            return {
                'method': 'tesseract',
                'text': full_text,
                'confidence': avg_confidence / 100,
                'words': extracted_text,
                'word_count': valid_words
            }
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {
                'method': 'tesseract',
                'text': '',
                'confidence': 0,
                'words': [],
                'word_count': 0,
                'error': str(e)
            }

    async def extract_text(self, image_bytes: bytes) -> Dict:
        """
        Extract text using the best method for the image
        """
        # Try layout-aware extraction first (better for tables)
        layout_result = await self.extract_text_with_layout(image_bytes)
        
        # If layout extraction has reasonable confidence, use it
        if layout_result.get('confidence', 0) > 0.3:
            logger.info(f"Using layout-aware extraction with confidence {layout_result['confidence']:.2f}")
            return layout_result
        
        # Otherwise fall back to standard extraction
        logger.info("Layout extraction failed, using standard OCR")
        return await self.extract_text_tesseract(image_bytes)