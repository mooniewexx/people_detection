"""
модуль для трекинга объектов между кадрами
"""

import numpy as np

class ObjectTracker:    
    """
    трекер для стабилизации детекций   
    """
    
    def __init__(self, max_age=10):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
    
    def update(self, detections):
        tracked_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = self._get_center(bbox)
            track_id = self._find_match(center)
            
            if track_id is not None:
                self.tracks[track_id] = {'center': center, 'age': 0}
                detection['track_id'] = track_id
            else:
                track_id = self.next_id
                self.tracks[track_id] = {'center': center, 'age': 0}
                detection['track_id'] = track_id
                self.next_id += 1
            
            tracked_detections.append(detection)
        
        self._clean_old_tracks()
        return tracked_detections
    
    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_match(self, center, max_dist=50):
        for track_id, track in self.tracks.items():
            track_center = track['center']
            dist = np.sqrt((center[0]-track_center[0])**2 + (center[1]-track_center[1])**2)
            if dist < max_dist:
                return track_id
        return None
    
    def _clean_old_tracks(self):
        to_remove = [tid for tid, track in self.tracks.items() if track['age'] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
        for track in self.tracks.values():
            track['age'] += 1
    
    def get_track_count(self):
        return len(self.tracks)

'''

class ObjectTracker:
    """
    трекер для стабилизации детекций между кадрами
    """
    
    def __init__(self, max_age = 10, distance_threshold = 50.0):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.distance_threshold = distance_threshold
    
    def update(self, detections):
        """
        обновление треков на основе новых детекций
        """
        tracked_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = self._get_center(bbox)
            
            # поиск ближайшего существующего трека
            track_id = self._find_closest_track(center)
            
            if track_id is not None:
                # обновление существующего трека
                self.tracks[track_id] = {
                    'center': center,
                    'age': 0
                }
                detection['track_id'] = track_id
            else:
                # создание нового трека
                track_id = self.next_id
                self.tracks[track_id] = {
                    'center': center,
                    'age': 0
                }
                detection['track_id'] = track_id
                self.next_id += 1
            
            tracked_detections.append(detection)
        
        # увеличение возраста и удаление старых треков
        self._age_and_clean_tracks()
        return tracked_detections
    
    def _get_center(self, bbox):
        """вычисление центра bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _find_closest_track(self, center):
        """поиск ближайшего трека"""
        closest_track_id = None
        min_distance = float('inf')
        
        for track_id, track in self.tracks.items():
            track_center = track['center']
            distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
            
            if distance < min_distance and distance < self.distance_threshold:
                min_distance = distance
                closest_track_id = track_id
        
        return closest_track_id
    
    def _age_and_clean_tracks(self):
        """увеличение возраста треков и удаление старых"""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] > self.max_age:
                tracks_to_remove.append(track_id)
        
        # удаляем старые
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def get_track_count(self):
        """получение количества активных треков"""
        return len(self.tracks)

        '''