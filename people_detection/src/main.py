"""
основной модуль пайплайна для детекции людей в видео.
объединяет: YOLOv8 (детекция), ObjectTracker (трекинг),
VideoProcessor (чтение/запись видео) и utils (визуализация, метрики).
"""

import cv2
import numpy as np
from src.detector import PeopleDetector
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.utils import (
    postprocessing,
    draw_detections,
    calculate_detection_metrics,
    print_metrics
)


class PeopleDetectionPipeline:
    """
    основной класс пайплайна для детекции людей на видео
    """

    def __init__(self, input_path: str, output_path: str,
                 model_name: str = "yolov8x.pt", conf_threshold: float = 0.3):
        """
        Args:
            input_path (str): путь к входному видеофайлу
            output_path (str): путь для сохранения выходного видео
            model_name (str): название предобученной модели YOLOv8
            conf_threshold (float): порог уверенности для детекции
        """
        self.detector = PeopleDetector(model_name, conf_threshold)
        self.tracker = ObjectTracker()
        self.video_processor = VideoProcessor(input_path, output_path)
        self.detections_per_frame = []
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'tracked_objects': set(),
            'confidence_history': []
        }

    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        обработка одного кадра: детекция, постобработка, трекинг, визуализация
        """
        detections = self.detector.detect_people(frame)
        detections = postprocessing(detections, frame.shape)
        detections = self.tracker.update(detections)
        self.detections_per_frame.append(detections)

        self.metrics['total_frames'] += 1
        self.metrics['total_detections'] += len(detections)
        for detection in detections:
            self.metrics['tracked_objects'].add(detection.get('track_id', -1))
            self.metrics['confidence_history'].append(detection['confidence'])

        processed_frame = draw_detections(frame, detections)

        info_text = [
            f"Frame: {frame_number}",
            f"People: {len(detections)}",
            f"Tracks: {self.tracker.get_track_count()}",
            f"Avg Conf: {np.mean([d['confidence'] for d in detections]):.2f}"
            if detections else "Avg Conf: 0.00"
        ]
        for i, text in enumerate(info_text):
            cv2.putText(
                processed_frame,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return processed_frame

    def run(self):
        """
        запуск обработки видео
        """
        self.video_processor.process_video(self.process_frame)
        metrics = calculate_detection_metrics(self.detections_per_frame)
        print_metrics(metrics)
        self.video_processor.release()
        print(f"\nобработка завершена. результат сохранён в: {self.video_processor.output_path}")