import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pymongo import MongoClient
import requests
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()

# Configuración de MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI")

DB_NAME = "Cluster0"
COLLECTION_NAME = "vehicles"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]




GOOGLE_API_KEY = os.getenv("API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  

class VehicleDetectionProcessor:
    def __init__(self, video_file, yolo_model_path="yolo12n.pt"):
        """Inicializa el procesador de detección de vehículos."""
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        try:
            self.yolo_model = YOLO(yolo_model_path)
            self.names = self.yolo_model.names
        except Exception as e:
            raise RuntimeError(f"Error cargando el modelo YOLO: {e}")

        self.cap = cv2.VideoCapture(video_file)
        if not self.cap.isOpened():
            raise FileNotFoundError("Error: No se pudo abrir el archivo de video.")

        self.area = np.array([(94, 363), (0,598), (1019, 599),(1019,360)], np.int32)
        self.processed_track_ids = set()
        self.cropped_images_folder = "cropped_vehicles"
        os.makedirs(self.cropped_images_folder, exist_ok=True)

    def analyze_image_with_gemini(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this image and extract only the following details:\n\n"
                     "|Vehicle Type | Vehicle Color | Vehicle Company |\n"
                     "|--------------|--------------|---------------|"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "description": "Detected vehicle"}
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()
        except Exception as e:
            print(f"Error invocando modelo Gemini: {e}")
            return "Error procesando imagen."

    def process_crop_image(self, image, track_id):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        image_filename = os.path.join(self.cropped_images_folder, f"vehicle_{track_id}.jpg")
        cv2.imwrite(image_filename, image)

        response_content = self.analyze_image_with_gemini(image_filename)
        extracted_data = response_content.split("\n")[2:]

        for row in extracted_data:
            if "--------------" in row or not row.strip():
                continue
            values = [col.strip() for col in row.split("|")[1:-1]]
            if len(values) == 3 and values != ["Vehicle Type", "Vehicle Color", "Vehicle Company"]:  
                vehicle_type, vehicle_color, vehicle_company = values
                data = {
                    "timestamp": timestamp,
                    "track_id": track_id,
                    "vehicle_type": vehicle_type,
                    "vehicle_color": vehicle_color,
                    "vehicle_company": vehicle_company
                }
                collection.insert_one(data)
                print(f"✅ Datos guardados en MongoDB Atlas para Track ID {track_id}.")

    def crop_and_process(self, frame, box, track_id):
        if track_id in self.processed_track_ids:
            return  

        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        if track_id not in self.processed_track_ids:
            self.processed_track_ids.add(track_id)
            threading.Thread(target=self.process_crop_image, args=(cropped_image, track_id), daemon=True).start()

    def process_video_frame(self, frame):
        frame = cv2.resize(frame, (1020, 600))
        results = self.yolo_model.track(frame, persist=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [-1] * len(boxes)
            allowed_classes = ["car", "truck"]
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                class_name = self.names[class_id]
                if class_name not in allowed_classes:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if cv2.pointPolygonTest(self.area, (x2, y2), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"ID: {track_id}", (x2, y2), 1, 1)
                    cvzone.putTextRect(frame, class_name, (x1, y1), 1, 1)
                    self.crop_and_process(frame, box, track_id)
        return frame
    
    def start_processing(self):
        cv2.namedWindow("Vehicle Detection")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_video_frame(frame)
            cv2.polylines(frame, [self.area], True, (0, 255, 0), 2)
            cv2.imshow("Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Procesamiento finalizado.")

if __name__ == "__main__":
    video_file = "video.mkv"
    processor = VehicleDetectionProcessor(video_file)
    processor.start_processing()
