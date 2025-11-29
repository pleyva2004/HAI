# S3/MinIO Image Storage Integration

## Overview

This document explains how to store and retrieve images (user-uploaded screenshots) using S3-compatible storage (MinIO for local development, AWS S3 for production).

**Why store images separately from the database?**
- ✅ Keeps database small and fast
- ✅ Cost-efficient (S3 storage is cheaper than database storage)
- ✅ Scalable (optimized for binary data)
- ✅ Industry standard approach

**Storage approach:**
- Images are uploaded to S3/MinIO
- Only the URL is stored in the database
- Images can be retrieved anytime using the URL

---

## Architecture

```
User uploads image (base64)
         ↓
Upload to S3/MinIO storage
         ↓
Get back URL: "s3://bucket/images/abc123.png"
         ↓
Store URL in Question.original_image_url
         ↓
[Later] Retrieve image using URL when needed
```

---

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# S3/MinIO Configuration
S3_ENDPOINT=http://localhost:9000              # MinIO for local dev
S3_ACCESS_KEY=minioadmin                       # MinIO default
S3_SECRET_KEY=minioadmin                       # MinIO default
S3_BUCKET_NAME=sat-questions                   # Bucket for question images
S3_REGION=us-east-1                            # AWS region (for production)

# For production (AWS S3):
# S3_ENDPOINT=https://s3.amazonaws.com
# S3_ACCESS_KEY=<your-aws-access-key>
# S3_SECRET_KEY=<your-aws-secret-key>
```

### Update `backend/config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# S3/MinIO Configuration
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "sat-questions")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
```

---

## Setup

### 1. Install Dependencies

Add to `backend/requirements.txt`:

```
boto3==1.34.0
```

Then install:

```bash
pip install boto3
```

### 2. Start MinIO (Local Development)

MinIO should already be in your `docker-compose.yml`. If not, add:

```yaml
services:
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

volumes:
  minio_data:
```

Start MinIO:

```bash
docker-compose up -d minio
```

Access MinIO console at: http://localhost:9001
- Username: `minioadmin`
- Password: `minioadmin`

### 3. Create Storage Bucket

Run this script once to create the bucket:

```bash
python scripts/setup_s3_bucket.py
```

---

## Implementation

### Create Storage Service

File: `backend/services/storage.py`

```python
"""
S3/MinIO storage service for handling image uploads and retrieval.
"""

import base64
import uuid
from datetime import datetime
from typing import Optional
import boto3
from botocore.exceptions import ClientError

from backend.config import (
    S3_ENDPOINT,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_BUCKET_NAME,
    S3_REGION
)


class StorageService:
    """
    Service for uploading and retrieving images from S3/MinIO storage.
    """

    def __init__(self):
        """Initialize S3 client"""
        self.s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION
        )
        self.bucket_name = S3_BUCKET_NAME

    def create_bucket_if_not_exists(self) -> None:
        """
        Create the S3 bucket if it doesn't already exist.

        Should be called during application startup.
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' already exists")
        except ClientError:
            # Bucket doesn't exist, create it
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                print(f"Created bucket '{self.bucket_name}'")
            except ClientError as e:
                print(f"Error creating bucket: {e}")
                raise

    def upload_image(
        self,
        image_base64: str,
        content_type: str = "image/png"
    ) -> str:
        """
        Upload a base64-encoded image to S3/MinIO.

        Args:
            image_base64: Base64 encoded image string
            content_type: MIME type of the image (default: image/png)

        Returns:
            URL to the uploaded image

        Example:
            storage = StorageService()
            url = storage.upload_image(base64_image)
            # url = "http://localhost:9000/sat-questions/images/20240115_abc123.png"
        """
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_base64)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"images/{timestamp}_{unique_id}.png"

        # Upload to S3/MinIO
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=image_bytes,
                ContentType=content_type
            )
        except ClientError as e:
            print(f"Error uploading image: {e}")
            raise

        # Return the URL
        url = f"{S3_ENDPOINT}/{self.bucket_name}/{filename}"
        return url

    def get_image(self, image_url: str) -> bytes:
        """
        Retrieve an image from S3/MinIO by URL.

        Args:
            image_url: Full URL to the image

        Returns:
            Image data as bytes

        Example:
            storage = StorageService()
            image_bytes = storage.get_image(question.original_image_url)

            # Save to file
            with open("image.png", "wb") as f:
                f.write(image_bytes)
        """
        # Extract bucket and key from URL
        # URL format: "http://localhost:9000/sat-questions/images/20240115_abc123.png"
        url_parts = image_url.replace(f"{S3_ENDPOINT}/", "").split("/", 1)
        bucket = url_parts[0]
        key = url_parts[1]

        # Download from S3/MinIO
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            image_bytes = response['Body'].read()
            return image_bytes
        except ClientError as e:
            print(f"Error retrieving image: {e}")
            raise

    def delete_image(self, image_url: str) -> None:
        """
        Delete an image from S3/MinIO by URL.

        Args:
            image_url: Full URL to the image

        Example:
            storage = StorageService()
            storage.delete_image(question.original_image_url)
        """
        # Extract bucket and key from URL
        url_parts = image_url.replace(f"{S3_ENDPOINT}/", "").split("/", 1)
        bucket = url_parts[0]
        key = url_parts[1]

        # Delete from S3/MinIO
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            print(f"Deleted image: {key}")
        except ClientError as e:
            print(f"Error deleting image: {e}")
            raise

    def image_exists(self, image_url: str) -> bool:
        """
        Check if an image exists in S3/MinIO.

        Args:
            image_url: Full URL to the image

        Returns:
            True if image exists, False otherwise
        """
        # Extract bucket and key from URL
        url_parts = image_url.replace(f"{S3_ENDPOINT}/", "").split("/", 1)
        bucket = url_parts[0]
        key = url_parts[1]

        # Check if object exists
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
```

---

## Workflow Integration

### 1. Update `extract_structure` Node

File: `backend/workflows/nodes.py`

```python
async def extract_structure_node(
    state: QuestionGenerationState,
    services: Dict[str, Any]
) -> QuestionGenerationState:
    """
    Extract structure from user input using Claude Vision.

    If user uploaded an image, save it to S3 and store the URL.
    """

    # Get services
    storage_service = services["storage"]
    ocr_service = services["ocr"]

    # If user uploaded an image
    if state.user_image:
        # Upload image to S3/MinIO and get URL
        image_url = storage_service.upload_image(state.user_image)
        print(f"Uploaded image to: {image_url}")

        # Store URL in state (you may want to add this field to QuestionGenerationState)
        # This URL will be copied to the generated question later
        state.original_image_url = image_url

        # Extract text using OCR
        extracted_text = await ocr_service.extract_text(state.user_image)
        state.extracted_text = extracted_text

    return state
```

### 2. Update `generate_question` Node

File: `backend/workflows/nodes.py`

```python
async def generate_question_node(
    state: QuestionGenerationState,
    services: Dict[str, Any]
) -> QuestionGenerationState:
    """
    Generate a new SAT question using LLM.
    """

    # ... your generation logic ...

    # Create question object
    new_question = Question(
        section=state.section or "Math",
        domain=state.predicted_domain or "Algebra",
        difficulty=state.predicted_difficulty or "Medium",
        skill=state.skill,
        question_text=generated_text,
        answer_choices=generated_choices,
        correct_answer=generated_answer,
        explanation=generated_explanation,
        original_image_url=getattr(state, 'original_image_url', None)  # Link to original!
    )

    state.generated_question = new_question
    return state
```

### 3. Initialize Services in Graph

File: `backend/workflows/graph.py`

```python
from backend.services.storage import StorageService
from backend.services.ocr_service import OCRService
# ... other imports ...

# Create service instances
services = {
    "storage": StorageService(),
    "ocr": OCRService(),
    # ... other services ...
}

# Ensure S3 bucket exists
services["storage"].create_bucket_if_not_exists()

# Pass services to workflow
result = workflow.invoke(initial_state, config={"services": services})
```

---

## Setup Script

File: `scripts/setup_s3_bucket.py`

```python
"""
Setup script to create S3/MinIO bucket for image storage.

Run once during initial setup:
    python scripts/setup_s3_bucket.py
"""

from backend.services.storage import StorageService


def main():
    """Create S3/MinIO bucket"""
    print("Setting up S3/MinIO storage...")

    storage = StorageService()
    storage.create_bucket_if_not_exists()

    print("✅ S3/MinIO setup complete!")


if __name__ == "__main__":
    main()
```

---

## Usage Examples

### Example 1: Upload Image from API

```python
from fastapi import FastAPI, UploadFile
from backend.services.storage import StorageService
import base64

app = FastAPI()
storage = StorageService()

@app.post("/api/upload-image")
async def upload_image(file: UploadFile):
    """Upload an image and return the URL"""

    # Read file bytes
    file_bytes = await file.read()

    # Encode to base64
    file_base64 = base64.b64encode(file_bytes).decode('utf-8')

    # Upload to S3/MinIO
    image_url = storage.upload_image(file_base64)

    return {"url": image_url}
```

### Example 2: Retrieve Image for Display

```python
from backend.services.storage import StorageService
from backend.database.queries import get_question_by_id

storage = StorageService()

# Get question from database
question = get_question_by_id("some-uuid")

# Retrieve original image
if question.original_image_url:
    image_bytes = storage.get_image(question.original_image_url)

    # Save to file
    with open("debug_original.png", "wb") as f:
        f.write(image_bytes)

    print(f"Saved original image to debug_original.png")
```

### Example 3: Check if Image Exists

```python
from backend.services.storage import StorageService

storage = StorageService()

url = "http://localhost:9000/sat-questions/images/20240115_abc123.png"

if storage.image_exists(url):
    print("✅ Image exists!")
else:
    print("❌ Image not found")
```

---

## Production Deployment (AWS S3)

When deploying to production, update your environment variables:

```bash
# Production .env
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=<your-aws-access-key-id>
S3_SECRET_KEY=<your-aws-secret-access-key>
S3_BUCKET_NAME=sat-questions-prod
S3_REGION=us-east-1
```

**No code changes needed!** The same code works with both MinIO and AWS S3.

### AWS S3 Setup Steps

1. Create S3 bucket in AWS console
2. Create IAM user with S3 permissions
3. Get access key and secret key
4. Update environment variables
5. Run the application

---

## Storage Costs

**Local Development (MinIO):**
- Free (runs in Docker)

**Production (AWS S3):**
- Storage: $0.023 per GB/month
- Uploads: $0.005 per 1,000 requests
- Downloads: $0.0004 per 1,000 requests

**Example cost for 10,000 questions (500KB each):**
- Storage: 5 GB × $0.023 = $0.12/month
- Very cost-effective! 💰

---

## Troubleshooting

### Error: "Bucket does not exist"

**Solution:** Run the setup script:
```bash
python scripts/setup_s3_bucket.py
```

### Error: "Access Denied"

**Solution:** Check your S3 credentials in `.env`:
- Verify `S3_ACCESS_KEY` and `S3_SECRET_KEY`
- For MinIO, default is `minioadmin` / `minioadmin`

### MinIO not starting

**Solution:** Check Docker:
```bash
docker-compose ps
docker-compose logs minio
```

### Images not displaying

**Solution:** Check the URL format:
- MinIO: `http://localhost:9000/bucket-name/path/to/image.png`
- AWS S3: `https://s3.amazonaws.com/bucket-name/path/to/image.png`

---

## Best Practices

1. **Always upload images before storing questions**
   - Upload → Get URL → Save URL to database

2. **Use descriptive filenames**
   - Include timestamp and UUID for uniqueness
   - Example: `images/20240115_143022_abc123.png`

3. **Handle upload failures gracefully**
   - Wrap uploads in try/except blocks
   - Return error to user if upload fails

4. **Don't delete images immediately**
   - Keep images even if question generation fails
   - Useful for debugging and retry logic

5. **Set up backup for production**
   - Enable S3 versioning
   - Configure S3 lifecycle policies for old images

---

## Summary

✅ Images stored in S3/MinIO (optimized for binary data)
✅ Only URLs stored in database (keeps DB fast and cheap)
✅ Easy to retrieve images anytime using URL
✅ Same code works for local (MinIO) and production (AWS S3)
✅ Cost-effective and scalable solution

Now you can efficiently store and retrieve images while keeping your database lean! 🎉
