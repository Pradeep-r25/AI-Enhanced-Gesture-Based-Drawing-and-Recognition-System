import cv2
import numpy as np
import argparse
from vision_integration import TextRecognizer


def main():
    parser = argparse.ArgumentParser(description='Test Google Vision API text recognition')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--mode', type=str, default='handwriting',
                        choices=['text', 'handwriting'],
                        help='Recognition mode: text or handwriting')
    args = parser.parse_args()

    if not args.image:
        print("Please provide an image path with --image")
        return

    # Initialize the text recognizer
    recognizer = TextRecognizer()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Could not load image: {args.image}")
        return

    # Resize if too large
    max_dim = 1500
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to make text clearer
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Perform recognition
    if args.mode == 'text':
        results = recognizer.recognize_text(thresh)
    else:
        results = recognizer.recognize_handwriting(thresh)

    # Display results
    print("\n===== Recognition Results =====")
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Recognized Text: {results['text']}")
        print(f"Confidence: {results.get('confidence', 0) * 100:.1f}%")

        if results.get('words'):
            print("\nIndividual Words:")
            for i, word in enumerate(results['words']):
                confidence = word.get('confidence', 0)
                print(f"  {i + 1}. '{word['text']}' (Confidence: {confidence * 100:.1f}%)")

    # Display the image with recognized text
    result_image = image.copy()

    # Draw bounding boxes for words
    for word in results.get('words', []):
        if 'bounding_box' in word:
            box = np.array(word['bounding_box'], dtype=np.int32)
            cv2.polylines(result_image, [box], True, (0, 255, 0), 2)

            # Add the recognized text above the bounding box
            x, y = box[0]
            cv2.putText(result_image, word['text'], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the image with annotations
    cv2.imshow("Recognition Results", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()