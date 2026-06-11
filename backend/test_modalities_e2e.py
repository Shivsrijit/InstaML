import requests
import time
import io
import wave
import numpy as np
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"

def generate_dummy_image():
    # Generate a simple 100x100 RGB image with PIL
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def generate_dummy_wav():
    # Generate a simple 1-second sine wave WAV file
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        # 44100 samples of quiet sound
        data = np.zeros(44100, dtype=np.int16)
        wav_file.writeframes(data.tobytes())
    buf.seek(0)
    return buf.getvalue()

def run_e2e_tests():
    print("Starting E2E Modality and Personalized Input Tests...")
    
    # 1. Register and Login
    auth_suffix = str(time.time()).replace(".", "")[-6:]
    username = f"user_{auth_suffix}"
    email = f"user_{auth_suffix}@instaml.com"
    password = "securepassword123"
    
    print(f"Registering user: {username}")
    reg_res = requests.post(f"{BASE_URL}/api/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    assert reg_res.status_code == 200, f"Registration failed: {reg_res.text}"
    
    print("Logging in...")
    login_res = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": username,
        "password": password
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # --- TEST 1: NLP - NER ---
    print("\n--- Test 1: NLP NER (Text) ---")
    print("Creating NER project...")
    proj_res = requests.post(f"{BASE_URL}/api/projects", json={
        "name": f"NER_Test_{auth_suffix}",
        "description": "NER project test",
        "data_type": "text",
        "task": "Named Entity Recognition (NER)"
    }, headers=headers)
    assert proj_res.status_code == 200, f"Create project failed: {proj_res.text}"
    project_id = proj_res.json()["id"]
    print(f"Project created with ID: {project_id}")
    
    print("Uploading raw text file...")
    raw_text = "Alice lives in London and works at Google. The date is Dec 25, 2026."
    text_file = io.BytesIO(raw_text.encode("utf-8"))
    upload_res = requests.post(
        f"{BASE_URL}/api/projects/{project_id}/upload",
        files={"file": ("sample.txt", text_file, "text/plain")},
        headers=headers
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    print("Upload successful!")
    
    print("Triggering training for NER pipeline...")
    train_res = requests.post(
        f"{BASE_URL}/api/projects/{project_id}/training/train",
        json={
            "model_name": "Pre-trained NER Pipeline",
            "target_col": "label",
            "use_hyperparameter_tuning": False,
            "validation_split": 0.2
        },
        headers=headers
    )
    assert train_res.status_code == 200, f"Train trigger failed: {train_res.text}"
    model_id = train_res.json()["id"]
    print(f"Model ID: {model_id}. Polling status...")
    
    for _ in range(10):
        status_res = requests.get(f"{BASE_URL}/api/projects/{project_id}/training/status", headers=headers)
        status = status_res.json()["status"]
        print(f"Training status: {status}")
        if status == "completed":
            break
        elif status == "failed":
            raise Exception(f"Training failed: {status_res.json()}")
        time.sleep(1)
    
    print("Deploying NER model...")
    deploy_res = requests.post(f"{BASE_URL}/api/projects/{project_id}/models/{model_id}/deploy", headers=headers)
    assert deploy_res.status_code == 200, f"Deploy failed: {deploy_res.text}"
    
    print("Making NER predictions...")
    pred_res = requests.post(
        f"{BASE_URL}/api/projects/{project_id}/deploy/predict",
        json={"features": {"text": "Emma works at Tesla in Paris."}},
        headers=headers
    )
    assert pred_res.status_code == 200, f"Predict failed: {pred_res.text}"
    pred_data = pred_res.json()
    print("NER Prediction output:", pred_data)
    assert "entities" in pred_data
    entities = pred_data["entities"]
    assert any(ent["label"] == "PERSON" and ent["text"] == "Emma" for ent in entities)
    assert any(ent["label"] == "ORGANIZATION" and ent["text"] == "Tesla" for ent in entities)
    assert any(ent["label"] == "LOCATION" and ent["text"] == "Paris" for ent in entities)
    print("NER E2E Test PASSED!")
    
    # --- TEST 2: Computer Vision - Face Detection ---
    print("\n--- Test 2: CV Face Detection (Image) ---")
    print("Creating Face Detection project...")
    proj_res = requests.post(f"{BASE_URL}/api/projects", json={
        "name": f"Face_Test_{auth_suffix}",
        "description": "Face Detection project test",
        "data_type": "image",
        "task": "Face Detection"
    }, headers=headers)
    assert proj_res.status_code == 200, f"Create project failed: {proj_res.text}"
    cv_project_id = proj_res.json()["id"]
    print(f"Project created with ID: {cv_project_id}")
    
    print("Uploading raw image file...")
    img_bytes = generate_dummy_image()
    img_file = io.BytesIO(img_bytes)
    upload_res = requests.post(
        f"{BASE_URL}/api/projects/{cv_project_id}/upload",
        files={"file": ("face.png", img_file, "image/png")},
        headers=headers
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    print("Upload successful!")
    
    print("Triggering training for Face Detection pipeline...")
    train_res = requests.post(
        f"{BASE_URL}/api/projects/{cv_project_id}/training/train",
        json={
            "model_name": "Haar Cascade Face Detector",
            "target_col": "dummy",
            "use_hyperparameter_tuning": False,
            "validation_split": 0.2
        },
        headers=headers
    )
    assert train_res.status_code == 200, f"Train trigger failed: {train_res.text}"
    model_id = train_res.json()["id"]
    print(f"Model ID: {model_id}. Polling status...")
    
    for _ in range(10):
        status_res = requests.get(f"{BASE_URL}/api/projects/{cv_project_id}/training/status", headers=headers)
        status = status_res.json()["status"]
        print(f"Training status: {status}")
        if status == "completed":
            break
        elif status == "failed":
            raise Exception(f"Training failed: {status_res.json()}")
        time.sleep(1)
        
    print("Deploying CV model...")
    deploy_res = requests.post(f"{BASE_URL}/api/projects/{cv_project_id}/models/{model_id}/deploy", headers=headers)
    assert deploy_res.status_code == 200, f"Deploy failed: {deploy_res.text}"
    
    print("Making file-based Face Detection prediction...")
    test_img = io.BytesIO(img_bytes)
    pred_res = requests.post(
        f"{BASE_URL}/api/projects/{cv_project_id}/deploy/predict-file",
        files={"file": ("test_face.png", test_img, "image/png")},
        headers=headers
    )
    assert pred_res.status_code == 200, f"Predict file failed: {pred_res.text}"
    pred_data = pred_res.json()
    print("Face Detection output:", pred_data.keys(), "Faces detected:", pred_data.get("faces_count"))
    assert "image" in pred_data
    assert "faces_count" in pred_data
    print("CV Face Detection E2E Test PASSED!")
    
    # --- TEST 3: Audio - Noise Reduction ---
    print("\n--- Test 3: Audio Noise Reduction ---")
    print("Creating Noise Reduction project...")
    proj_res = requests.post(f"{BASE_URL}/api/projects", json={
        "name": f"Audio_Test_{auth_suffix}",
        "description": "Noise Reduction project test",
        "data_type": "audio",
        "task": "Noise Reduction"
    }, headers=headers)
    assert proj_res.status_code == 200, f"Create project failed: {proj_res.text}"
    audio_project_id = proj_res.json()["id"]
    print(f"Project created with ID: {audio_project_id}")
    
    print("Uploading raw WAV audio file...")
    wav_bytes = generate_dummy_wav()
    wav_file = io.BytesIO(wav_bytes)
    upload_res = requests.post(
        f"{BASE_URL}/api/projects/{audio_project_id}/upload",
        files={"file": ("audio.wav", wav_file, "audio/wav")},
        headers=headers
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    print("Upload successful!")
    
    print("Triggering training for Noise Reduction pipeline...")
    train_res = requests.post(
        f"{BASE_URL}/api/projects/{audio_project_id}/training/train",
        json={
            "model_name": "Low-Pass DSP Denoising Filter",
            "target_col": "dummy",
            "use_hyperparameter_tuning": False,
            "validation_split": 0.2
        },
        headers=headers
    )
    assert train_res.status_code == 200, f"Train trigger failed: {train_res.text}"
    model_id = train_res.json()["id"]
    print(f"Model ID: {model_id}. Polling status...")
    
    for _ in range(10):
        status_res = requests.get(f"{BASE_URL}/api/projects/{audio_project_id}/training/status", headers=headers)
        status = status_res.json()["status"]
        print(f"Training status: {status}")
        if status == "completed":
            break
        elif status == "failed":
            raise Exception(f"Training failed: {status_res.json()}")
        time.sleep(1)
        
    print("Deploying Audio model...")
    deploy_res = requests.post(f"{BASE_URL}/api/projects/{audio_project_id}/models/{model_id}/deploy", headers=headers)
    assert deploy_res.status_code == 200, f"Deploy failed: {deploy_res.text}"
    
    print("Making file-based Noise Reduction prediction...")
    test_wav = io.BytesIO(wav_bytes)
    pred_res = requests.post(
        f"{BASE_URL}/api/projects/{audio_project_id}/deploy/predict-file",
        files={"file": ("test_audio.wav", test_wav, "audio/wav")},
        headers=headers
    )
    assert pred_res.status_code == 200, f"Predict file failed: {pred_res.text}"
    pred_data = pred_res.json()
    print("Audio Noise Reduction output:", pred_data.keys())
    assert "audio" in pred_data
    print("Audio Noise Reduction E2E Test PASSED!")
    
    print("\nALL E2E MODALITY AND PERSONALIZED INPUT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_e2e_tests()
