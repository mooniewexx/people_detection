"""
модуль для детекции объектов с YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch


class PeopleDetector:
    """
    класс для детекции людей на изображениях и видео
    YOLOv8 для высокой точности распознавания
    """
    
    def __init__(self, model_name = 'yolov8x.pt', conf_threshold = 0.25):
        """
        инициализация детектора
        
        Args:
            model_name (str): название модели
            conf_threshold (float): порог уверенности для детекции
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """загрузка предобученной модели"""
        try:
            self.model = YOLO(self.model_name)
            print(f"модель {self.model_name} успешно загружена")
            print(f"используемое устройство: {self.device}")
        except Exception as e:
            raise RuntimeError(f"ошибка загрузки модели: {e}")
    
    def detect_people(self, image):
        """
        детекция людей на изображении
        
        Args:
            image (np.ndarray): входное изображение в формате BGR
            
        Returns:
            List[Dict]: список детектированных людей с координатами и уверенностью
        """
        if self.model is None:
            raise RuntimeError("модель не загружена")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.model(
            image_rgb, 
            conf=self.conf_threshold, 
            iou=0.4,
            imgsz=640,
            augment=True,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls) == 0 and box.conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf.cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': 'person'
                        })
        
        return detections
    
    def get_model_info(self):
        """получение информации о модели"""
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.conf_threshold,
            'device': self.device
        }