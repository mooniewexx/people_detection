"""
вспомогательные функции для визуализации и обработки
"""

import cv2
import numpy as np
import random

np.random.seed(42)
random.seed(42)

color_map = {}

def get_color_for_id(track_id):
    """
    возвращает уникальный и стабильный цвет для данного track_id
    если цвета ещё нет — создаёт его
    """
    if track_id not in color_map:
        color_map[track_id] = (
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255)),
            int(np.random.randint(50, 255))
        )
    return color_map[track_id]

def postprocessing(detections, frame_shape):
    """
    пост-обработка детекций
    
    Args:
        detections (List[Dict]): список детекций
        frame_shape (Tuple[int, int]): размер кадра (height, width)
        
    Returns:
        List[Dict]: отфильтрованные детекции
    """
    filtered_detections = []
    frame_height, frame_width = frame_shape[:2]
    frame_area = frame_width * frame_height
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        
        if bbox_area < frame_area * 0.001:
            continue
            
        if bbox_area > frame_area * 0.3:
            continue
            
        aspect_ratio = bbox_height / bbox_width if bbox_width > 0 else 0
        if not (1.5 < aspect_ratio < 5.0):
            continue
        
        filtered_detections.append(detection)
    
    return filtered_detections

def calculate_visible_ratio(bbox, frame_width, frame_height):
    """
    расчет доли bbox, которая видна в кадре
    
    Args:
        bbox (List[int]): bbox [x1, y1, x2, y2]
        frame_width (int): ширина кадра
        frame_height (int): высота кадра
        
    Returns:
        float: доля видимой области (0.0 - 1.0)
    """
    x1, y1, x2, y2 = bbox
    
    visible_x1 = max(0, x1)
    visible_y1 = max(0, y1)
    visible_x2 = min(frame_width, x2)
    visible_y2 = min(frame_height, y2)
    
    bbox_area = (x2 - x1) * (y2 - y1)
    visible_area = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
    
    return visible_area / bbox_area if bbox_area > 0 else 0.0

def draw_detections(image, 
                    detections, 
                    confidence_threshold = 0.25):
    """
    отрисовка детекций на изображении
    
    Args:
        image (np.ndarray): исходное изображение
        detections (List[Dict]): список детекций
        confidence_threshold (float): порог для отображения уверенности
        
    Returns:
        np.ndarray: изображение с отрисованными детекциями
    """
    result_image = image.copy()
    
    for detection in detections:
        if detection['confidence'] >= confidence_threshold:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            track_id = detection.get('track_id', -1)

            color = get_color_for_id(track_id)
            
            thickness = max(1, min(3, int(min(x2-x1, y2-y1) * 0.01)))
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            if track_id != -1:
                label = f"person ID:{track_id} {confidence:.2f}"
            else:
                label = f"person {confidence:.2f}"
            
            font_scale = max(0.4, min(0.8, (x2-x1) * 0.002))
            thickness_text = max(1, int(font_scale * 2))
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text
            )
            
            cv2.rectangle(
                result_image, 
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            cv2.putText(
                result_image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255), 
                thickness_text
            )
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(result_image, (center_x, center_y), 3, color, -1)
    
    return result_image

def calculate_detection_metrics(detections_per_frame):
    """
    pасчет метрик детекции
    
    Args:
        detections_per_frame (List[List[Dict]]): детекции по кадрам
        
    Returns:
        Dict: метрики детекции
    """
    total_detections = 0
    total_frames = len(detections_per_frame)
    confidences = []
    track_ids = set()
    
    for frame_detections in detections_per_frame:
        total_detections += len(frame_detections)
        for detection in frame_detections:
            confidences.append(detection['confidence'])
            if 'track_id' in detection:
                track_ids.add(detection['track_id'])
    
    metrics = {
        'total_frames': total_frames,
        'total_detections': total_detections,
        'unique_tracks': len(track_ids),
        'average_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0,
        'average_confidence': np.mean(confidences) if confidences else 0,
        'min_confidence': min(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0,
        'confidence_std': np.std(confidences) if confidences else 0
    }
    
    return metrics

def print_metrics(metrics):
    print("\nметрики детекции:")
    print(f"всего кадров: {metrics['total_frames']}")
    print(f"всего детекций: {metrics['total_detections']}")
    print(f"уникальных треков: {metrics['unique_tracks']}")
    print(f"среднее детекций на кадр: {metrics['average_detections_per_frame']:.2f}")
    print(f"средняя уверенность: {metrics['average_confidence']:.3f}")
    print(f"min уверенность: {metrics['min_confidence']:.3f}")
    print(f"max уверенность: {metrics['max_confidence']:.3f}")