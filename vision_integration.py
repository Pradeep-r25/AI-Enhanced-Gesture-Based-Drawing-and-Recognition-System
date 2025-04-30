from google.cloud import vision
import io
import cv2
import numpy as np


class TextRecognizer:
    def __init__(self):
        """Initialize the Vision API client"""
        self.client = vision.ImageAnnotatorClient()

    def recognize_text(self, image):
        """
        Recognize text in the provided image using Google Vision API

        Args:
            image: A numpy array containing the image data

        Returns:
            A dictionary with recognized text and confidence scores
        """
        # Convert the numpy array to bytes
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            return {"error": "Failed to encode image"}

        content = encoded_image.tobytes()

        image = vision.Image(content=content)

        # Perform text detection
        response = self.client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            return {"text": "", "words": [], "confidence": 0}

        # Extract the full text (first element contains all text)
        full_text = texts[0].description if texts else ""

        # Extract individual words and their bounding boxes
        words = []
        for text in texts[1:]:  # Skip the first one as it contains all text
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            words.append({
                "text": text.description,
                "bounding_box": vertices
            })

        if response.error.message:
            return {"error": response.error.message}

        return {
            "text": full_text,
            "words": words,
            "confidence": 0.9  # Vision API doesn't provide confidence scores directly
        }

    def recognize_handwriting(self, image):
        """
        Specifically recognize handwritten text (better for cursive)

        Args:
            image: A numpy array containing the image data

        Returns:
            A dictionary with recognized text and confidence scores
        """
        # Convert the numpy array to bytes
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            return {"error": "Failed to encode image"}

        content = encoded_image.tobytes()

        image = vision.Image(content=content)

        # Use document_text_detection which is better for handwriting
        response = self.client.document_text_detection(image=image)

        # Process the response
        result = {"text": "", "words": [], "confidence": 0}

        if response.full_text_annotation:
            result["text"] = response.full_text_annotation.text

            # Extract words with confidence scores
            words = []
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = ''.join([symbol.text for symbol in word.symbols])
                            confidence = word.confidence

                            # Get bounding box
                            vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]

                            words.append({
                                "text": word_text,
                                "confidence": confidence,
                                "bounding_box": vertices
                            })

            result["words"] = words

            # Average confidence across all words
            if words:
                result["confidence"] = sum(word["confidence"] for word in words) / len(words)

        if response.error.message:
            result["error"] = response.error.message

        return result