"""
точка входа в проект. запускает пайплайн детекции людей
"""

import os
from src.main import PeopleDetectionPipeline

def main():
    input_video = "crowd.mp4"
    output_video = "output/crowd_detected.mp4"

    if not os.path.exists(input_video):
        print(f"ошибка: входной файл '{input_video}' не найден.")
        return

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    pipeline = PeopleDetectionPipeline(
        input_path=input_video,
        output_path=output_video,
        model_name="yolov8x.pt",
        conf_threshold=0.3
    )

    try:
        pipeline.run()
    except Exception as e:
        print(f"произошла ошибка во время обработки: {e}")


if __name__ == "__main__":
    main()
