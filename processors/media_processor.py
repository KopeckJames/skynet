# processors/media_processor.py
import base64
import tempfile
import os
from typing import Dict, Any
import pytesseract
from PIL import Image
import cv2
from moviepy.editor import VideoFileClip
import openai
from .base_processor import BaseProcessor
from ..config import Config

class MediaProcessor(BaseProcessor):
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.openai_client = openai.OpenAI(api_key=openai_api_key)

    def process(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process media content based on type"""
        if not self.validate_content(content):
            raise ValueError("Invalid content provided")

        content_type = content.get("type")
        file_path = content.get("path")

        if content_type.startswith("image/"):
            return self.process_image(file_path)
        elif content_type.startswith("audio/"):
            return self.process_audio(file_path)
        elif content_type.startswith("video/"):
            return self.process_video(file_path)
        else:
            raise ValueError(f"Unsupported media type: {content_type}")

    def process_image(self, file_path: str) -> Dict[str, str]:
        """Process image using OCR and GPT-4-Vision"""
        try:
            image = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(image)
            
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return {
                "ocr_text": ocr_text,
                "description": response.choices[0].message.content
            }
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def process_audio(self, file_path: str) -> Dict[str, str]:
        """Process audio using OpenAI Whisper API"""
        try:
            with open(file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return {
                "transcript": transcript.text,
                "type": "audio_transcript"
            }
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

    def process_video(self, file_path: str) -> Dict[str, str]:
        """Process video - extract audio and analyze frames"""
        try:
            # Extract audio and transcribe
            video = VideoFileClip(file_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                video.audio.write_audiofile(temp_audio.name)
                audio_result = self.process_audio(temp_audio.name)
                os.unlink(temp_audio.name)

            # Process video frames
            cap = cv2.VideoCapture(file_path)
            frames = []
            frame_descriptions = []
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // 5)  # Extract 5 frames
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0 and len(frames) < 5:
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Process extracted frames
            for frame in frames:
                with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_frame:
                    cv2.imwrite(temp_frame.name, frame)
                    frame_info = self.process_image(temp_frame.name)
                    frame_descriptions.append(frame_info["description"])
            
            return {
                "transcript": audio_result["transcript"],
                "frame_descriptions": frame_descriptions,
                "type": "video_analysis"
            }
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")