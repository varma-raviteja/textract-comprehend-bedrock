import os
import boto3
from tqdm import tqdm
from PIL import Image

def get_image_files(directory):
    """Get all jpg and png files in the given directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

def should_process_file(file_path):
    """Check if the file should be processed (i.e., no corresponding txt file exists)."""
    txt_path = os.path.splitext(file_path)[0] + '.txt'
    return not os.path.exists(txt_path)

def extract_text_from_image(image_path):
    """Extract text from the image using Amazon Textract."""
    textract_client = boto3.client('textract')
    
    with open(image_path, 'rb') as image:
        response = textract_client.detect_document_text(Document={'Bytes': image.read()})
    
    extracted_text = []
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            extracted_text.append(item['Text'])
    
    return '\n'.join(extracted_text)

def summarize_text(text):
    """Summarize the extracted text using Amazon Comprehend."""
    comprehend_client = boto3.client('comprehend')
    
    if len(text) > 5000:
        text = text[:5000]  # Amazon Comprehend has a limit of 5000 bytes per document

    key_phrases_response = comprehend_client.detect_key_phrases(Text=text, LanguageCode='en')
    key_phrases = [phrase['Text'] for phrase in key_phrases_response['KeyPhrases']]

    sentiment_response = comprehend_client.detect_sentiment(Text=text, LanguageCode='en')
    sentiment = sentiment_response['Sentiment']

    summary = "Summary:\n" + '\n'.join(key_phrases[:5])  # Limiting to top 5 key phrases
    summary += f"\n\nSentiment: {sentiment}"

    return summary

def save_text_to_file(text, file_path):
    """Save the extracted text to a file."""
    txt_path = os.path.splitext(file_path)[0] + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

def save_summary_to_file(summary, file_path):
    """Save the summary to a file with a '_summary' suffix."""
    summary_path = os.path.splitext(file_path)[0] + '_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

def process_images_in_directory(directory):
    """Process all images in the given directory."""
    image_files = get_image_files(directory)
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(directory, image_file)
        
        if should_process_file(image_path):
            extracted_text = extract_text_from_image(image_path)
            save_text_to_file(extracted_text, image_path)
            
            summary = summarize_text(extracted_text)
            save_summary_to_file(summary, image_path)

# Usage in Jupyter notebook or standalone script
directory = '.'  # Current directory
process_images_in_directory(directory)