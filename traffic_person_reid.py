import sys
import os
import pickle
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import torch
from torchvision import transforms
from PIL import Image
import cv2
from ultralytics import YOLO
from model import ReIDNet

MAX_RESULTS = 5
YOLO_MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5

class PersonRetrievalSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 加载模型权重，自动获取类别数
        state_dict = torch.load("best_model.pth", map_location=self.device)
        num_classes = state_dict['fc.weight'].shape[0]
        print(f"从模型权重中读取类别数: {num_classes}")

        self.model = ReIDNet(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.yolo = YOLO(YOLO_MODEL_PATH)
        if self.device == "cuda":
            self.yolo.to("cuda")

        # 缓存相关
        self.gallery_feats = None
        self.gallery_names = None
        self.current_gallery_path = None

    def _load_cache(self, gallery_path):
        """加载指定图库路径下的特征缓存"""
        cache_file = os.path.join(gallery_path, "gallery_features.pkl")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(
                f"未找到特征缓存文件: {cache_file}\n"
                f"请先运行 extract_gallery_features.py 并指定图库路径为: {gallery_path}"
            )
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        self.gallery_feats = data["feats"]
        self.gallery_names = data["names"]
        self.current_gallery_path = gallery_path
        print(f"已加载图库缓存: {len(self.gallery_names)} 张图片 (路径: {gallery_path})")

    def _extract_feature(self, img_or_path):
        try:
            if isinstance(img_or_path, str):
                img = Image.open(img_or_path).convert("RGB")
            elif isinstance(img_or_path, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img_or_path, cv2.COLOR_BGR2RGB))
            else:
                img = img_or_path
            x = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, feat = self.model(x)
            return feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None

    def detect_persons(self, img_path_or_array):
        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array)
        else:
            img = img_path_or_array
        results = self.yolo(img, classes=0)
        persons = []
        boxes = []
        for r in results:
            for box in r.boxes:
                if box.conf.item() >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        persons.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                        boxes.append((x1, y1, x2, y2))
        return persons, boxes

    def search_similar_person(self, query_img, gallery_path="gallery", top_k=MAX_RESULTS,
                              is_person_crop=False, exclude_frame=None):
        """
        检索相似行人
        :param query_img: 查询图片（PIL Image 或路径）
        :param gallery_path: 图库目录路径（支持绝对路径）
        :param top_k: 返回结果数量
        :param is_person_crop: 是否直接传入裁剪好的行人图片
        :param exclude_frame: 排除的帧名（用于过滤同帧）
        :return: list of (图片路径, 相似度)
        """
        # 如果图库路径改变，重新加载缓存
        if self.current_gallery_path != gallery_path:
            self._load_cache(gallery_path)

        if is_person_crop and isinstance(query_img, Image.Image):
            q_feat = self._extract_feature(query_img)
        else:
            persons, _ = self.detect_persons(query_img)
            if persons:
                q_feat = self._extract_feature(persons[0])
            else:
                q_feat = self._extract_feature(query_img)

        if q_feat is None or self.gallery_feats is None:
            return []

        q_norm = q_feat / (np.linalg.norm(q_feat) + 1e-8)
        gallery_norms = np.linalg.norm(self.gallery_feats, axis=1, keepdims=True)
        gallery_normed = self.gallery_feats / (gallery_norms + 1e-8)
        scores = np.dot(gallery_normed, q_norm)

        sorted_indices = np.argsort(scores)[::-1]
        results = []
        for idx in sorted_indices:
            name = self.gallery_names[idx]
            if exclude_frame and exclude_frame in name:
                continue
            full_path = os.path.join(gallery_path, name)
            results.append((full_path, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

_system = None
def get_system():
    global _system
    if _system is None:
        _system = PersonRetrievalSystem()
    return _system

class CameraWorker(QThread):
    frame_signal = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_signal.emit(q_img.scaled(220, 300, Qt.AspectRatioMode.KeepAspectRatio))
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.system = get_system()
        self.camera_worker = None
        self.temp_capture_path = "temp_capture.jpg"
        self.use_yolo = True
        self.query_img_path = ""
        self.library_path = "gallery"   # 默认图库路径
        self.match_res = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("行人重识别系统 (PRW + YOLOv8 + 多目标)")
        self.resize(1000, 750)
        w = QWidget()
        layout = QVBoxLayout(w)
        self.setCentralWidget(w)

        self.btn_local = QPushButton("1. 选择本地图片")
        self.btn_cam = QPushButton("2. 打开实时摄像头")
        self.btn_cap = QPushButton("3. 抓拍当前行人")
        
        # 自定义图库路径输入
        self.lbl_lib = QLabel("图库路径:")
        self.txt_lib = QLineEdit("gallery")
        self.btn_lib = QPushButton("4. 设置图库路径")
        
        self.chk_yolo = QCheckBox("启用YOLO行人检测 (自动检测多目标)")
        self.chk_yolo.setChecked(True)
        self.chk_yolo.toggled.connect(self.toggle_yolo)
        self.btn_search = QPushButton("5. 开始匹配检索 (自动过滤同帧)")
        self.btn_export = QPushButton("6. 导出Excel结果")

        self.img_show = QLabel()
        self.img_show.setFixedSize(220, 300)
        self.img_show.setStyleSheet("border:1px solid #cccccc;")

        # 使用列表显示缩略图
        self.result_list = QListWidget()
        self.result_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.result_list.setIconSize(QSize(80, 80))
        self.result_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.result_list.setGridSize(QSize(180, 140))
        self.result_list.setWordWrap(True)
        self.result_list.setFlow(QListWidget.Flow.LeftToRight)
        self.result_list.setFixedHeight(350)

        # 布局
        layout.addWidget(self.btn_local)
        layout.addWidget(self.btn_cam)
        layout.addWidget(self.btn_cap)
        layout.addWidget(self.img_show)
        # 图库路径设置行
        lib_layout = QHBoxLayout()
        lib_layout.addWidget(self.lbl_lib)
        lib_layout.addWidget(self.txt_lib)
        lib_layout.addWidget(self.btn_lib)
        layout.addLayout(lib_layout)
        layout.addWidget(self.chk_yolo)
        layout.addWidget(self.btn_search)
        layout.addWidget(QLabel("检索匹配结果（多目标分组显示，已排除同帧干扰）"))
        layout.addWidget(self.result_list)
        layout.addWidget(self.btn_export)

        self.btn_local.clicked.connect(self.choose_local_img)
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_cap.clicked.connect(self.capture_person)
        self.btn_lib.clicked.connect(self.set_library_path)
        self.btn_search.clicked.connect(self.do_search)
        self.btn_export.clicked.connect(self.export_excel)

    def toggle_yolo(self, checked):
        self.use_yolo = checked

    def set_library_path(self):
        path = self.txt_lib.text().strip()
        if path:
            self.library_path = path
            QMessageBox.information(self, "设置成功", f"图库路径已设置为: {path}")

    def open_camera(self):
        self.camera_worker = CameraWorker()
        self.camera_worker.frame_signal.connect(self.update_cam_img)
        self.camera_worker.start()
        QMessageBox.information(self, "提示", "摄像头已开启")

    def update_cam_img(self, qimg):
        self.img_show.setPixmap(QPixmap.fromImage(qimg))

    def capture_person(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(self.temp_capture_path, frame)
            self.query_img_path = self.temp_capture_path
            self.img_show.setPixmap(QPixmap(self.temp_capture_path).scaled(220,300,Qt.AspectRatioMode.KeepAspectRatio))
            QMessageBox.information(self, "抓拍成功", "已抓拍")
        cap.release()

    def choose_local_img(self):
        path, _ = QFileDialog.getOpenFileName(filter="图片(*.jpg *.png *.jpeg)")
        if path:
            self.query_img_path = path
            self.img_show.setPixmap(QPixmap(path).scaled(220,300,Qt.AspectRatioMode.KeepAspectRatio))

    def do_search(self):
        if not self.query_img_path:
            QMessageBox.warning(self, "缺少条件", "请先选择或抓拍图片")
            return

        self.result_list.clear()
        self.match_res = []

        query_basename = os.path.basename(self.query_img_path)
        frame_name = os.path.splitext(query_basename)[0]

        if self.use_yolo:
            persons, _ = self.system.detect_persons(self.query_img_path)
            if not persons:
                QMessageBox.warning(self, "未检测到行人", "将使用整图检索（仍排除同帧）")
                res = self.system.search_similar_person(
                    self.query_img_path, gallery_path=self.library_path,
                    is_person_crop=False, exclude_frame=frame_name
                )
                self._display_group("整图检索", res)
                self.match_res = self._format_results(res, "整图")
            else:
                total_res = []
                for idx, person_img in enumerate(persons):
                    res = self.system.search_similar_person(
                        person_img, gallery_path=self.library_path,
                        is_person_crop=True, top_k=MAX_RESULTS,
                        exclude_frame=frame_name
                    )
                    self._display_group(f"行人 {idx+1}", res)
                    total_res.extend(self._format_results(res, f"行人 {idx+1}"))
                self.match_res = total_res
                QMessageBox.information(self, "检测结果", f"共检测到 {len(persons)} 个行人（已排除同帧干扰）")
        else:
            res = self.system.search_similar_person(
                self.query_img_path, gallery_path=self.library_path,
                is_person_crop=False, exclude_frame=frame_name
            )
            self._display_group("整图检索", res)
            self.match_res = self._format_results(res, "整图")

        QMessageBox.information(self, "完成", f"返回 {len(self.match_res)} 个结果")

    def _display_group(self, title, results):
        # 添加分组标题
        title_item = QListWidgetItem(f"————— {title} —————")
        title_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.result_list.addItem(title_item)

        for idx, (img_p, sim) in enumerate(results):
            text = f"{idx+1}. {os.path.basename(img_p)}\n相似度: {sim:.4f}"
            item = QListWidgetItem(text)
            # 加载缩略图
            pixmap = QPixmap(img_p)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(pixmap))
            else:
                item.setIcon(QIcon())
            self.result_list.addItem(item)

    def _format_results(self, results, group_name):
        return [{
            "行人编号": group_name,
            "排名": i+1,
            "匹配图片": os.path.basename(p),
            "相似度": s
        } for i, (p, s) in enumerate(results)]

    def export_excel(self):
        if not self.match_res:
            QMessageBox.warning(self, "无数据", "暂无结果")
            return
        df = pd.DataFrame(self.match_res)
        df.to_excel("行人检索结果_多目标.xlsx", index=False)
        QMessageBox.information(self, "导出成功", "已保存为 行人检索结果_多目标.xlsx")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())