"""
модуль для обработки видеофайлов
"""

import cv2
from tqdm import tqdm
import os


class VideoProcessor:
    """
    класс для обработки видео: чтение, обработка кадров, запись результата
    """
    
    def __init__(self, input_path, output_path):
        """
        инициализация видео процессора
        """
        self.input_path = input_path
        self.output_path = output_path
        self.cap = None
        self.out = None
        self._setup_video_io()
    
    def _setup_video_io(self):
        """настройка видео ввода/вывода"""
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"не удалось открыть видеофайл: {self.input_path}")
        
        # получение параметров видео
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # настройка кодеков для кроссплатформенности
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # создание папки для выходного файла если необходимо
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        self.out = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        if not self.out.isOpened():
            raise RuntimeError(f"не удалось инициализировать видео writer: {self.output_path}")
    
    def process_video(self, 
                     process_frame_func, 
                     progress_callback = None):
        """
        обработка видео кадр за кадром
        
        Args:
            process_frame_func (Callable): функция для обработки кадра
            progress_callback (Callable, optional): функция для отслеживания прогресса
        """
        frame_count = 0
        
        # прогресс бар
        pbar = tqdm(total=self.total_frames, desc="обработка видео")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # обработка кадра
            processed_frame = process_frame_func(frame, frame_count)
            
            # запись
            self.out.write(processed_frame)
            
            frame_count += 1
            pbar.update(1)
            
            if progress_callback:
                progress_callback(frame_count, self.total_frames)

        self.release()
        pbar.close()        
    
    def get_video_info(self):
        """получение информации о видео"""
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'duration_seconds': self.total_frames / self.fps if self.fps > 0 else 0
        }
    
    def release(self):
        """освобождение ресурсов"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()