"""
модуль для детекции объектов с YOLOv8
"""

import cv2
import tqdm
import numpy as np
from ultralytics import YOLO
from src.tracker import ObjectTracker
from src.utils import get_color_for_id


class PeopleDetection:
    """
    пайплайн для детекции людей в видео.
    """
    
    def __init__(self,input_video_path,output_video_path, model_name = 'yolov8x.pt', confidence_threshold = 0.25):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        self.model = YOLO(model_name)
        self.tracker = ObjectTracker(max_age=10)
        
        # статистика
        self.all_detections = []
        self.metrics = {
            'total_frames': 0,
            'total_detections': 0,
            'tracked_objects': set(),
            'confidence_history': []
        }
    
    def detect_people(self, image):
        """
        детекция людей на изображении
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.model(
            image_rgb, 
            conf=self.confidence_threshold, 
            iou=0.4,
            imgsz=640,
            augment=True,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # только люди (class_id = 0 в сосо)
                    if int(box.cls) == 0 and box.conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf.cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': 'person'
                        })
        
        return detections
    
    def postprocessing(self, detections, frame_shape):
        filtered_detections = []
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_width * frame_height
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # по bounding box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area = bbox_width * bbox_height
            
            # слишком маленькие детекции
            if bbox_area < frame_area * 0.001:  # < 0.1% кадра
                continue
                
            # слишком большие детекции (мб ложные)
            if bbox_area > frame_area * 0.3: 
                continue
                
            # по соотношению сторон 
            aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
            if not (1.5 < aspect_ratio < 5.0):  # типичное для людей
                continue
            
            filtered_detections.append(detection)
        
        return filtered_detections
    
    def draw_detections(self, image, detections):
        """
        отрисовка детекций на изображении
        """
        
        result_image = image.copy()
        
        for detection in detections:
            if detection['confidence'] >= self.confidence_threshold:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                track_id = detection.get('track_id', -1)

                color = get_color_for_id(track_id)
                
                # толщина рамки
                thickness = max(1, min(3, int(min(x2-x1, y2-y1) * 0.01)))
                
                # отрисовка bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), thickness)
                
                if track_id != -1:
                    label = f"person ID:{track_id} {confidence:.2f}"
                else:
                    label = f"person {confidence:.2f}"
                
                font_scale = max(0.4, min(0.8, (x2-x1) * 0.002))
                thickness_text = max(1, int(font_scale * 2))
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text
                )
                
                # фон для текста
                cv2.rectangle(
                    result_image, 
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    -1
                )
                
                cv2.putText(result_image, label,
                    (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness_text)
                
                # отрисовка центра объекта
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(result_image, (center_x, center_y), 3, -1)
        
        return result_image
    
    def process_frame(self, frame, frame_number):
        """
        улучшенная обработка одного кадра
        """
        # детекция людей
        raw_detections = self.detect_people(frame)
        # пост-обработка
        filtered_detections = self.postprocessing(raw_detections, frame.shape)
        # трекинг объектов
        tracked_detections = self.tracker.update(filtered_detections)
        # сохранение для метрик
        self.all_detections.append(tracked_detections)
    
        self.metrics['total_frames'] += 1
        self.metrics['total_detections'] += len(tracked_detections)
        for detection in tracked_detections:
            self.metrics['tracked_objects'].add(detection.get('track_id', -1))
            self.metrics['confidence_history'].append(detection['confidence'])
        
        # отрисовка детекций
        processed_frame = self.draw_detections(frame, tracked_detections)
        
        info_text = [
            f"Frame: {frame_number}",
            f"People: {len(tracked_detections)}",
            f"Tracks: {self.tracker.get_track_count()}",
            f"Avg Conf: {np.mean([d['confidence'] for d in tracked_detections]):.2f}" if tracked_detections else "Avg Conf: 0.00"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame,text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return processed_frame
    
    def process_video(self):
        """обработка видео"""
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"не удалось открыть файл: {self.input_video_path}")
        
        # параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"видео: {width}x{height}, {fps} fps, {total_frames} кадров")
        print(f"модель: {self.model_name}")
        print(f"порог уверенности: {self.confidence_threshold}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        pbar = tqdm(total=total_frames, desc="обработка видео")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # обработка кадра
            processed_frame = self.process_frame(frame, frame_count)
            # запись
            out.write(processed_frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        return frame_count
    
    def calculate_metrics(self):
        """расчет метрик детекции"""
        total_detections = self.metrics['total_detections']
        total_frames = self.metrics['total_frames']
        confidences = self.metrics['confidence_history']
        
        metrics = {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'unique_tracks': len(self.metrics['tracked_objects']),
            'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
            'average_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """печать метрик"""
        print("\nметрики детекции:")
        print(f"всего кадров: {metrics['total_frames']}")
        print(f"всего детекций: {metrics['total_detections']}")
        print(f"уникальных треков: {metrics['unique_tracks']}")
        print(f"среднее детекций на кадр: {metrics['average_detections_per_frame']:.2f}")
        print(f"средняя уверенность: {metrics['average_confidence']:.3f}")
        print(f"min уверенность: {metrics['min_confidence']:.3f}")
        print(f"max уверенность: {metrics['max_confidence']:.3f}")
    
    def run(self):
        """запуск пайплайна"""
        total_frames = self.process_video()
        metrics = self.calculate_metrics()
        self.print_metrics(metrics)
        
        print(f"\nобработка завершена, результат сохранен в: {self.output_video_path}")
        
        return metrics




'''
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
        
        # детекция
        results = self.model(
            image_rgb, 
            conf=self.conf_threshold, 
            iou=0.4,
            imgsz=640,
            augment=True,
            agnostic_nms=False,
            max_det=100,
            verbose=False
        )
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # только люди (class_id = 0 в COCO)
                    if int(box.cls) == 0 and box.conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf.cpu().numpy())
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_name': 'person'
                        })
        
        return detections
    
    def multi_scale_detection(self, image, scales = [0.5, 1.0, 1.5]):
        """
        детекция на разных масштабах для лучшего обнаружения
        
        Args:
            image (np.ndarray): входное изображение
            scales (List[float]): список масштабов для детекции
            
        Returns:
            List[Dict]: объединенные детекции со всех масштабов
        """
        all_detections = []
        original_height, original_width = image.shape[:2]
        
        for scale in scales:
            # изменение размера изображения
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # детекция на измененном размере
            scale_detections = self.detect_people(resized_image)
            
            # масштабирование координат обратно
            for detection in scale_detections:
                x1, y1, x2, y2 = detection['bbox']
                detection['bbox'] = [
                    int(x1 / scale), int(y1 / scale),
                    int(x2 / scale), int(y2 / scale)
                ]
            
            all_detections.extend(scale_detections)
        
        # удаление дубликатов
        return self._remove_duplicate_detections(all_detections)
    
    def _remove_duplicate_detections(self, detections, iou_threshold = 0.5):
        """
        удаление дублирующихся детекций с помощью NMS
        
        Args:
            detections (List[Dict]): список детекций
            iou_threshold (float): порог IoU для удаления дубликатов
            
        Returns:
            List[Dict]: отфильтрованные детекции
        """
        if not detections:
            return []
        
        # сортировка по уверенности
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        used_indices = set()
        
        for i in range(len(detections)):
            if i in used_indices:
                continue
                
            current_det = detections[i]
            filtered_detections.append(current_det)
            used_indices.add(i)
            
            for j in range(i + 1, len(detections)):
                if j in used_indices:
                    continue
                    
                other_det = detections[j]
                iou = self._calculate_iou(current_det['bbox'], other_det['bbox'])
                
                if iou > iou_threshold:
                    used_indices.add(j)
        
        return filtered_detections
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        расчет IoU (Intersection over Union) между двумя bbox
        
        Args:
            bbox1 (List[int]): первый bounding box [x1, y1, x2, y2]
            bbox2 (List[int]): второй bounding box [x1, y1, x2, y2]
            
        Returns:
            float: значение IoU
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # вычисление площади пересечения
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # вычисление площадей bboxes
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # вычисление IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    
    def get_model_info(self):
        """получение информации о модели"""
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.conf_threshold,
            'device': self.device
        }
        '''

'''
