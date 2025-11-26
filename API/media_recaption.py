import os
import json
import sys
import cv2
import shutil
import traceback
from PIL import Image
from pathlib import Path
from collections import defaultdict
import time
import openai
import base64
import numpy as np
from io import BytesIO

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

class MediaProcessor:
    def __init__(self, 
                 input_dir: str,
                 root_dir: str,
                 prompt_text: str = "",
                 annotation_type: str = "recaption",
                 language: str = "英文",
                 fps: float = 1.0,
                 api_key: str,
                 api_base: str,
                 model: str = "Qwen2.5-VL-3B-Instruct"):
        """
        初始化媒体处理器
        :param api_key: API密钥
        :param api_base: API基础地址
        :param input_dir: 输入文件目录
        :param root_dir: 输出根目录
        :param model: 使用的模型名称
        :param language: 处理语言
        :param fps: 视频帧提取帧率
        :param prompt_text: 提示文本
        :param annotation_type: 要添加的标注类型
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.input_dir = input_dir
        self.root_dir = root_dir
        self.model = model
        self.language = language
        self.fps = fps
        self.prompt_text = prompt_text + f"important: 请使用{self.language}输出，输出不要分段落，按照一整段输出。"
        self.annotation_type = annotation_type
        
        # 确保输出目录存在
        os.makedirs(self.root_dir, exist_ok=True)
        
        # 初始化元数据
        self.ensure_metadata_exists()

    def check_environment(self):
        return {
            "python_path": sys.executable,
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None')
        }

    def get_file_type(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in VIDEO_EXTENSIONS:
            return "video"
        return None

    def extract_video_frames(self, video_path, fps=2.0):
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开文件: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(video_fps / fps) if video_fps > fps else 1
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            count += 1

        cap.release()
        if not frames:
            raise ValueError("无法从视频中提取任何帧")
        return frames, video_fps

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def media_recaption(self, image_path=None, video_path=None, frames=None, language="英文"):
        try:
          messages = [
              {"role": "system", "content": "you are a helpful assisstant."}
          ]
  
          user_content = []
  
          # 添加文本提示
          user_content.append({"type": "text", "text": self.prompt_text})
  
          # 添加图像或视频
          if image_path:
              if not os.path.exists(image_path):
                  raise FileNotFoundError(f"图像文件不存在：{image_path}")
              base64_image = self.encode_image(image_path)
              user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
          elif video_path or frames:
              if not frames:
                  frames, video_fps = self.extract_video_frames(video_path, fps=self.fps)
  
              base64_frames = []
              for frame in frames:
                  buffer = BytesIO()
                  frame.save(buffer, format="jpeg")
                  base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
  
              user_content.append({
                  "type": "video_url",
                  "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
              })
          else:
              raise ValueError("必须提供图像路径、视频路径或视频帧")
  
          messages.append({"role": "user", "content": user_content})
  
          # 调用API
          response = self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              extra_body={"mm_processor_kwargs": {"fps": [1.0]}} if (video_path or frames) else {}
          )
  
          return response.choices[0].message.content
  
      except Exception as e:
          raise RuntimeError(f"实体识别失败: {str(e)}")

    def copy_and_rename_media_file(self, source_path, dest_dir, file_id):
        ext = os.path.splitext(source_path)[1].lower()
        dest_path = os.path.join(dest_dir, f"{file_id}{ext}")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return dest_path

    def get_max_id_in_dir(self, root_dir, file_type):
        if file_type == "image":
            search_dir = os.path.join(root_dir, "labels")
        elif file_type == "video":
            search_dir = os.path.join(root_dir, "annotations")
        else:
            return -1
    
        max_id = -1
    
        # 遍历所有分桶目录
        if os.path.exists(search_dir):
            for bucket_dir_name in os.listdir(search_dir):
                bucket_path = os.path.join(search_dir, bucket_dir_name)
                if os.path.isdir(bucket_path):
                    # 遍历分桶目录下的所有.json文件
                    for filename in os.listdir(bucket_path):
                        if filename.endswith('.json'):
                            file_id_str = os.path.splitext(filename)[0]
                            if file_id_str.isdigit():
                                current_id = int(file_id_str)
                                if current_id > max_id:
                                    max_id = current_id
    
        return max_id

    def find_annotation_file(self, file_path, root_dir):
        file_type = self.get_file_type(file_path)
        if not file_type:
            return None
    
        # 检查文件是否已经在输出目录中（即是否是之前处理过的文件）
        # 如果是，直接用其文件名作为file_id
        if file_path.startswith(os.path.join(root_dir, "images")) or file_path.startswith(os.path.join(root_dir, "videos")):
            file_name = os.path.basename(file_path)
            file_id = os.path.splitext(file_name)[0]
            if file_id.isdigit():
                bucket = f"{int(file_id) // 1000:03d}"
                if file_type == "image":
                    anno_path = os.path.join(root_dir, "labels", bucket, f"{file_id}.json")
                else:
                    anno_path = os.path.join(root_dir, "annotations", bucket, f"{file_id}.json")
                if os.path.exists(anno_path):
                    return anno_path
    
        # 否则，按原始文件名查找（适用于处理已有标注的数据集）
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
    
        # 增加健壮性检查：确保 file_id 是有效的整数
        if not file_id.isdigit():
            # 如果文件名不是数字，则无法按照现有规则查找标注文件
            return None
    
        bucket = f"{int(file_id) // 1000:03d}"
    
        # 标注文件可能存在的位置
        possible_paths = []
        if file_type == "image":
            possible_paths.append(os.path.join(root_dir, "labels", bucket, f"{file_id}.json"))
        else:  # video
            possible_paths.append(os.path.join(root_dir, "annotations", bucket, f"{file_id}.json"))
    
        # 检查可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
        return None

    def create_new_annotation(self, file_path, root_dir, total_count):
        try:
            file_type = self.get_file_type(file_path)
            if not file_type:
                raise ValueError(f"不支持的文件类型: {file_path}")

            file_id = f"{total_count:05d}"
            bucket = f"{int(file_id) // 1000:03d}"

            # 复制媒体文件到输出目录
            if file_type == "image":
                media_dest_dir = os.path.join(root_dir, "images", bucket)
                new_media_path = self.copy_and_rename_media_file(file_path, media_dest_dir, file_id)
                anno_dest_dir = os.path.join(root_dir, "labels", bucket)
                with Image.open(new_media_path) as img:
                    width, height = img.size
                    mode = img.mode

                anno = {
                    "image_id": file_id,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "source_url": "",
                    "annotations": {
                        "caption": "",
                        "spatial": [],
                        "relations": [],
                        "scene_graph": {},
                        self.annotation_type: []  # 使用指定的标注类型
                    }
                }
            else:  # video
                media_dest_dir = os.path.join(root_dir, "videos", bucket)
                new_media_path = self.copy_and_rename_media_file(file_path, media_dest_dir, file_id)
                anno_dest_dir = os.path.join(root_dir, "annotations", bucket)
                cap = cv2.VideoCapture(new_media_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / video_fps if video_fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                anno = {
                    "video_id": file_id,
                    "duration": round(duration, 2),
                    "fps": round(video_fps, 2),
                    "width": width,
                    "height": height,
                    "source_url": "",
                    "annotations": {
                        "caption": "",
                        "temporal": [],
                        "events": [],
                        "camera_motion": {},
                        "scene_graph_sequence": [],
                        self.annotation_type: []  # 使用指定的标注类型
                    }
                }

            os.makedirs(anno_dest_dir, exist_ok=True)
            anno_path = os.path.join(anno_dest_dir, f"{file_id}.json")

            with open(anno_path, 'w', encoding='utf-8') as f:
                json.dump(anno, f, ensure_ascii=False, indent=2)

            return anno_path, file_id, new_media_path

        except Exception as e:
            raise RuntimeError(f"创建新标注文件失败: {str(e)}")

    def ensure_metadata_exists(self):
        """确保元数据文件存在，如果不存在则创建"""
        # 确保meta_data目录存在
        meta_dir = os.path.join(self.root_dir, "meta_data")
        os.makedirs(meta_dir, exist_ok=True)

        # 确保images和videos目录存在
        os.makedirs(os.path.join(self.root_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "videos"), exist_ok=True)

        # 检查并创建图片元数据文件
        image_meta_path = os.path.join(self.root_dir, "metadata.json")
        if not os.path.exists(image_meta_path):
            supported_annotations = ["caption", "spatial", "relations", "scene_graph"]
            if self.annotation_type not in supported_annotations:
                supported_annotations.append(self.annotation_type)
                
            image_meta = {
                "dataset": {
                    "name": "custom_image_dataset",
                    "type": "image_captioning",
                    "path": str(Path(self.root_dir).resolve()),
                    "image_dir": "images/",
                    "annotation_dir": "labels/",
                    "metadata_file": "metadata.json"
                },
                "data_stats": {
                    "total_images": 0,
                    "total_storage_size_mb": 0,
                    "avg_file_size_mb": 0,
                    "image_formats": {},
                    "resolution_distribution": {},
                    "mode_distribution": {}
                },
                "data_format": {
                    "image_extensions": list(IMAGE_EXTENSIONS),
                    "supported_annotations": supported_annotations
                },
                "processing_info": {
                    "generated_captions": True,
                    "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
            }
            with open(image_meta_path, 'w', encoding='utf-8') as f:
                json.dump(image_meta, f, ensure_ascii=False, indent=2)

        # 检查并创建视频元数据文件
        video_meta_path = os.path.join(meta_dir, "stats.json")
        if not os.path.exists(video_meta_path):
            supported_annotations = ["caption", "temporal", "events", "camera_motion"]
            if self.annotation_type not in supported_annotations:
                supported_annotations.append(self.annotation_type)
                
            video_meta = {
                "dataset": {
                    "name": "custom_video_dataset",
                    "type": "video_captioning",
                    "path": str(Path(self.root_dir).resolve()),
                    "video_dir": "videos/",
                    "annotation_dir": "annotations/",
                    "metadata_file": "meta_data/stats.json"
                },
                "data_stats": {
                    "total_videos": 0,
                    "total_duration_seconds": 0,
                    "total_duration_hours": 0,
                    "avg_video_length_seconds": 0,
                    "video_formats": {},
                    "resolution_distribution": {},
                    "fps_distribution": {},
                    "total_storage_size_gb": 0,
                    "avg_file_size_gb": 0
                },
                "data_format": {
                    "video_extensions": list(VIDEO_EXTENSIONS),
                    "supported_annotations": supported_annotations
                },
                "processing_info": {
                    "generated_captions": True,
                    "extracted_frames": True,
                    "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                }
            }
            with open(video_meta_path, 'w', encoding='utf-8') as f:
                json.dump(video_meta, f, ensure_ascii=False, indent=2)

        # 检查并创建splits文件
        splits_path = os.path.join(meta_dir, "splits.json")
        if not os.path.exists(splits_path):
            splits = {
                "train": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
                "validation": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
                "test": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""}
            }
            with open(splits_path, 'w', encoding='utf-8') as f:
                json.dump(splits, f, ensure_ascii=False, indent=2)

    def update_metadata_supported_annotations(self):
        """更新元数据文件，添加标注类型到supported_annotations"""
        # 更新图片元数据
        image_meta_path = os.path.join(self.root_dir, "metadata.json")
        if os.path.exists(image_meta_path):
            with open(image_meta_path, 'r', encoding='utf-8') as f:
                image_meta = json.load(f)

            if self.annotation_type not in image_meta["data_format"]["supported_annotations"]:
                image_meta["data_format"]["supported_annotations"].append(self.annotation_type)
                image_meta["processing_info"]["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                with open(image_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(image_meta, f, ensure_ascii=False, indent=2)

        # 更新视频元数据
        video_meta_path = os.path.join(self.root_dir, "meta_data", "stats.json")
        if os.path.exists(video_meta_path):
            with open(video_meta_path, 'r', encoding='utf-8') as f:
                video_meta = json.load(f)

            if self.annotation_type not in video_meta["data_format"]["supported_annotations"]:
                video_meta["data_format"]["supported_annotations"].append(self.annotation_type)
                video_meta["processing_info"]["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                with open(video_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(video_meta, f, ensure_ascii=False, indent=2)

    def update_metadata_statistics(self):
        """根据处理后的实际数据，更新元数据文件中的统计信息"""
        # 更新图片元数据 (metadata.json)
        image_meta_path = os.path.join(self.root_dir, "metadata.json")
        if os.path.exists(image_meta_path):
            with open(image_meta_path, 'r', encoding='utf-8') as f:
                image_meta = json.load(f)

            image_dir = os.path.join(self.root_dir, "images")
            total_images = 0
            total_storage_size_mb = 0
            image_formats = {}
            resolution_distribution = {}

            if os.path.exists(image_dir):
                for bucket_dir in Path(image_dir).rglob('*'):
                    if bucket_dir.is_dir():
                        for image_file in bucket_dir.iterdir():
                            if image_file.is_file() and self.get_file_type(str(image_file)) == "image":
                                total_images += 1

                                # 计算文件大小
                                size_mb = image_file.stat().st_size / (1024 * 1024)
                                total_storage_size_mb += size_mb

                                # 统计文件格式
                                ext = image_file.suffix.lower()
                                image_formats[ext] = image_formats.get(ext, 0) + 1

                                # 统计分辨率
                                try:
                                    with Image.open(image_file) as img:
                                        res_key = f"{img.width}x{img.height}"
                                        resolution_distribution[res_key] = resolution_distribution.get(res_key, 0) + 1
                                except Exception:
                                    pass

            image_meta["data_stats"]["total_images"] = total_images
            image_meta["data_stats"]["total_storage_size_mb"] = round(total_storage_size_mb, 2)
            image_meta["data_stats"]["avg_file_size_mb"] = round(total_storage_size_mb / total_images, 2) if total_images > 0 else 0
            image_meta["data_stats"]["image_formats"] = image_formats
            image_meta["data_stats"]["resolution_distribution"] = resolution_distribution

            with open(image_meta_path, 'w', encoding='utf-8') as f:
                json.dump(image_meta, f, ensure_ascii=False, indent=2)

        # 更新视频元数据 (meta_data/stats.json)
        video_meta_path = os.path.join(self.root_dir, "meta_data", "stats.json")
        if os.path.exists(video_meta_path):
            with open(video_meta_path, 'r', encoding='utf-8') as f:
                video_meta = json.load(f)

            video_dir = os.path.join(self.root_dir, "videos")
            total_videos = 0
            total_duration_seconds = 0
            total_storage_size_gb = 0
            video_formats = {}
            resolution_distribution = {}
            fps_distribution = {}

            if os.path.exists(video_dir):
                for bucket_dir in Path(video_dir).rglob('*'):
                    if bucket_dir.is_dir():
                        for video_file in bucket_dir.iterdir():
                            if video_file.is_file() and self.get_file_type(str(video_file)) == "video":
                                total_videos += 1

                                # 计算文件大小
                                size_gb = video_file.stat().st_size / (1024 * 1024 * 1024)
                                total_storage_size_gb += size_gb

                                # 统计文件格式
                                ext = video_file.suffix.lower()
                                video_formats[ext] = video_formats.get(ext, 0) + 1

                                # 统计视频信息
                                try:
                                    cap = cv2.VideoCapture(str(video_file))
                                    if cap.isOpened():
                                        fps = cap.get(cv2.CAP_PROP_FPS)
                                        fps_key = f"{round(fps, 1)}"
                                        fps_distribution[fps_key] = fps_distribution.get(fps_key, 0) + 1

                                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        res_key = f"{width}x{height}"
                                        resolution_distribution[res_key] = resolution_distribution.get(res_key, 0) + 1

                                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                        duration = frame_count / fps if fps > 0 else 0
                                        total_duration_seconds += duration
                                    cap.release()
                                except Exception:
                                    pass

            video_meta["data_stats"]["total_videos"] = total_videos
            video_meta["data_stats"]["total_duration_seconds"] = round(total_duration_seconds, 2)
            video_meta["data_stats"]["total_duration_hours"] = round(total_duration_seconds / 3600, 2)
            video_meta["data_stats"]["avg_video_length_seconds"] = round(total_duration_seconds / total_videos, 2) if total_videos > 0 else 0
            video_meta["data_stats"]["total_storage_size_gb"] = round(total_storage_size_gb, 2)
            video_meta["data_stats"]["avg_file_size_gb"] = round(total_storage_size_gb / total_videos, 2) if total_videos > 0 else 0
            video_meta["data_stats"]["video_formats"] = video_formats
            video_meta["data_stats"]["resolution_distribution"] = resolution_distribution
            video_meta["data_stats"]["fps_distribution"] = fps_distribution

            with open(video_meta_path, 'w', encoding='utf-8') as f:
                json.dump(video_meta, f, ensure_ascii=False, indent=2)

    def process_media_file(self, file_path, idx):
        """处理单个媒体文件，识别实体并更新标注"""
        try:
            file_type = self.get_file_type(file_path)
            if not file_type:
                return {
                    "success": False,
                    "path": file_path,
                    "error": "不支持的文件类型"
                }

            # 查找或创建标注文件
            anno_path = self.find_annotation_file(file_path, self.root_dir)
            if not anno_path:
                # 创建新的标注文件，并复制媒体文件到输出目录
                max_id = self.get_max_id_in_dir(self.root_dir, file_type)
                new_id = max_id + 1
                anno_path, file_id, new_media_path = self.create_new_annotation(file_path, self.root_dir, new_id)
                process_path = new_media_path
            else:
                file_id = os.path.splitext(os.path.basename(anno_path))[0]
                process_path = file_path

            # 加载现有标注
            with open(anno_path, 'r', encoding='utf-8') as f:
                annotation_data = json.load(f)

            # 识别实体
            if file_type == "image":
                recaption_result = self.media_recaption(
                    image_path=process_path,
                    language=self.language
                )
            else:  # video
                frames, _ = self.extract_video_frames(process_path, fps=self.fps)
                recaption_result = self.media_recaption(
                    frames=frames,
                    language=self.language
                )

            # 更新标注中的实体信息
            if "annotations" not in annotation_data:
                annotation_data["annotations"] = {}

            annotation_data["annotations"][self.annotation_type] = recaption_result

            # 保存更新后的标注
            with open(anno_path, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, ensure_ascii=False, indent=2)

            return {
                "success": True,
                "id": file_id,
                "original_path": file_path,
                "processed_path": process_path,
                "annotation_path": anno_path,
                "type": file_type
            }

        except Exception as e:
            return {
                "success": False,
                "path": file_path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def collect_files(self):
        """收集目录中的所有图像和视频文件"""
        files = []
        for root, _, file_names in os.walk(self.input_dir):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                file_type = self.get_file_type(file_path)
                if file_type:
                    files.append((file_path, file_type))
        return files

    def process(self):
        """执行批量处理"""
        # 初始化结果字典
        result = {
            "success": False,
            "root_dir": self.root_dir,
            "environment_info": self.check_environment(),
            "error": None,
            "details": {
                "processed_files": [],
                "errors": []
            }
        }

        try:
            if not os.path.exists(self.input_dir):
                raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")

            # 收集文件
            files = self.collect_files()
            if not files:
                raise ValueError(f"在输入目录中未找到任何图像或视频文件: {self.input_dir}")

            # 处理文件
            processed = []
            for idx, (file_path, file_type) in enumerate(files):
                print(f"正在处理 {idx + 1}/{len(files)}: {file_path}")
                res = self.process_media_file(file_path, idx)
                res["type"] = file_type
                processed.append(res)

                # 打印处理结果摘要
                if res["success"]:
                    print(f"处理成功 -> 生成标注 ID: {res['id']}")
                else:
                    print(f"处理失败: {res['error']}")

            # 更新结果统计
            result["success"] = all(item["success"] for item in processed)
            result["details"]["processed_files"] = [
                {
                    "id": item["id"],
                    "original_path": item["original_path"],
                    "processed_path": item["processed_path"],
                    "annotation_path": item["annotation_path"],
                    "type": item["type"]
                }
                for item in processed if item["success"]
            ]
            result["details"]["errors"] = [
                {
                    "path": item["path"],
                    "error": item["error"],
                    "type": item["type"]
                }
                for item in processed if not item["success"]
            ]

            # 更新元数据
            self.update_metadata_supported_annotations()
            self.update_metadata_statistics()

            print(f"\n处理完成 - 成功: {len(result['details']['processed_files'])}, 失败: {len(result['details']['errors'])}")
            print(f"结果摘要已保存至: {os.path.join(self.root_dir, 'recaption_processing_result.json')}")

        except Exception as e:
            result["error"] = str(e)
            print(f"\n处理过程中发生严重错误: {str(e)}", file=sys.stderr)
            traceback.print_exc()

        # 保存处理结果摘要
        with open(os.path.join(self.root_dir, "recaption_processing_result.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        return result

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Media Recaption Tool - Generate captions for images/videos")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input media files")
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory for output results")
    parser.add_argument("--prompt-text", type=str, default="识别物体并计数。", help="Additional prompt text for caption generation")
    parser.add_argument("--annotation-type", type=str, default="recaption", help="Type of annotation to add (default: recaption)")
    parser.add_argument("--language", type=str, default="英文", help="Output language (default: 英文)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate for videos (default: 1.0)")
    parser.add_argument("--api-key", type=str, default="key", help="API key for model access")
    parser.add_argument("--api-base", type=str, default="http://10.59.67.2:5018/v1", help="Base URL for API endpoint")
    parser.add_argument("--model", type=str, default="Qwen3-VL", help="model name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    # 初始化媒体处理器
    processor = MediaProcessor(
        input_dir=args.input_dir,
        root_dir=args.root_dir,
        prompt_text=args.prompt_text,
        annotation_type=args.annotation_type,
        language=args.language,
        fps=args.fps,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.model
    )
    # 这里可以添加实际处理逻辑（例如遍历输入目录并处理文件）
    print("Media processor initialized successfully.")
    # 示例：打印环境信息
    print("Environment info:", processor.check_environment())
    print("processing...")
    processor.process()
    print("Media processor process successfully!")

if __name__ == "__main__":
    main()
