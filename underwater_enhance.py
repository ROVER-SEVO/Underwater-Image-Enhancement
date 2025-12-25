import sys
import cv2
import numpy as np
import os

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, 
                             QGroupBox, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


# Core Image Processing Logic
class ImageProcessor:
    @staticmethod
    def white_balance(img, a_shift, b_shift):
        """Corrects color cast in LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        A = A.astype(np.float32)
        B = B.astype(np.float32)

        # Shift A (Green-Red) and B (Blue-Yellow) components
        A = A - (np.mean(A) - 128) + a_shift
        B = B - (np.mean(B) - 128) + b_shift

        A = np.clip(A, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def restore_red(img, strength):
        """Compensates for red light absorption."""
        b, g, r = cv2.split(img)
        # Histogram equalization on the red channel
        boost = cv2.equalizeHist(r)
        strength_factor = strength / 100.0
        # Blend original red with boosted red
        r_new = cv2.addWeighted(r, 1 - strength_factor, boost, strength_factor, 0)
        return cv2.merge([b, g, r_new])

    @staticmethod
    def clahe_enhance(img, clip_limit):
        """Contrast-Limited Adaptive Histogram Equalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        # Limit contrast enhancement to avoid amplifying noise
        clip_val = max(clip_limit / 10.0, 0.1)
        clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def dehaze(img, omega):
        """Dehazing using Dark Channel Prior."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Estimate atmospheric light (top 5% brightest pixels)
        A_light = np.percentile(gray, 95)
        if A_light == 0: A_light = 1
        
        omega_val = omega / 100.0
        t = 1 - omega_val * (gray / A_light)
        t = np.clip(t, 0.35, 1.0) # T_MIN to prevent over-enhancement
        
        t_stack = cv2.merge([t, t, t])
        J = (img.astype(np.float32) - A_light) / t_stack + A_light
        return np.clip(J, 0, 255).astype(np.uint8)

    @staticmethod
    def sharpen(img):
        """Adaptive Unsharp Masking."""
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        return cv2.addWeighted(img, 1.2, blur, -0.2, 0)

    @staticmethod
    def gamma_correct(img, g=1.1):
        """Gamma correction to brighten shadows."""
        inv = 1.0 / g
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    @staticmethod
    def process_pipeline(img, params):
        """Executes the full enhancement pipeline."""
        if img is None: return None
        
        res = ImageProcessor.white_balance(img, params['a_shift'], params['b_shift'])
        res = ImageProcessor.restore_red(res, params['red_strength'])
        res = ImageProcessor.clahe_enhance(res, params['clahe_clip'])
        res = ImageProcessor.dehaze(res, params['omega'])
        res = ImageProcessor.sharpen(res)
        res = ImageProcessor.gamma_correct(res)
        return res


#PyQt5 Main Window (GUI)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCV Underwater Image Enhancer")
        self.resize(1200, 800) 

        # Data storage
        self.original_image = None  # Full resolution original
        self.preview_image = None   # Downscaled preview
        self.processed_image = None # Processed preview

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        
        controls_layout = QVBoxLayout()
        controls_group = QGroupBox("Parameters")
        controls_group.setFixedWidth(300)
        controls_inner_layout = QVBoxLayout()

        self.sliders = {}
        # Config: (key, min, max, default, label)
        slider_configs = [
            ('a_shift', -50, 50, 0, "A Shift (Green-Red)"),
            ('b_shift', -50, 50, 0, "B Shift (Blue-Yellow)"),
            ('omega', 0, 100, 75, "Dehaze (Omega)"),
            ('clahe_clip', 1, 50, 12, "Contrast (CLAHE)"),
            ('red_strength', 0, 100, 30, "Red Boost")
        ]

        for key, min_v, max_v, def_v, label_text in slider_configs:
            lbl = QLabel(f"{label_text}: {def_v}")
            sld = QSlider(Qt.Horizontal)
            sld.setMinimum(min_v)
            sld.setMaximum(max_v)
            sld.setValue(def_v)
            # Connect signal
            sld.valueChanged.connect(lambda val, k=key, l=lbl, txt=label_text: self.on_slider_change(val, k, l, txt))
            
            controls_inner_layout.addWidget(lbl)
            controls_inner_layout.addWidget(sld)
            self.sliders[key] = sld

        controls_inner_layout.addStretch()
        controls_group.setLayout(controls_inner_layout)
        
        # Buttons
        btn_layout = QVBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_save = QPushButton("Save Result")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_save)

        controls_layout.addWidget(controls_group)
        controls_layout.addLayout(btn_layout)
        controls_layout.addStretch()

        # Right: Image Display
        display_layout = QGridLayout()
        
        self.lbl_orig_title = QLabel("Original")
        self.lbl_orig_title.setAlignment(Qt.AlignCenter)
        self.lbl_orig_title.setMaximumHeight(30)
        
        self.lbl_proc_title = QLabel("Enhanced")
        self.lbl_proc_title.setAlignment(Qt.AlignCenter)
        self.lbl_proc_title.setMaximumHeight(30)

        # Original Image Label
        self.lbl_orig_img = QLabel()
        self.lbl_orig_img.setAlignment(Qt.AlignCenter)
        self.lbl_orig_img.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        # FIX: Set Policy to Ignored to prevent resizing loop
        self.lbl_orig_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # Processed Image Label
        self.lbl_proc_img = QLabel()
        self.lbl_proc_img.setAlignment(Qt.AlignCenter)
        self.lbl_proc_img.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        # FIX: Set Policy to Ignored
        self.lbl_proc_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        # Add to layout (row, col)
        display_layout.addWidget(self.lbl_orig_title, 0, 0)
        display_layout.addWidget(self.lbl_proc_title, 0, 1)
        display_layout.addWidget(self.lbl_orig_img, 1, 0)
        display_layout.addWidget(self.lbl_proc_img, 1, 1)
        
        # Let image row stretch to fill vertical space
        display_layout.setRowStretch(1, 1)

        
        main_layout.addLayout(controls_layout)
        main_layout.addLayout(display_layout, stretch=1)

    def on_slider_change(self, value, key, label_obj, label_text):
        """Handle slider value changes."""
        display_val = value
        if key == 'omega': display_val = value / 100.0
        if key == 'clahe_clip': display_val = value / 10.0
        
        label_obj.setText(f"{label_text}: {display_val}")
        # Reprocess only if an image is loaded
        if self.original_image is not None:
            self.update_processing()

    def get_current_params(self):
        return {k: v.value() for k, v in self.sliders.items()}

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", 
                                                   "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)", options=options)
        if file_path:
            # Read image (supports non-ASCII/Unicode paths)
            img_data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is None:
                return

            self.original_image = img
            
            # Create downscaled preview for performance (max width 1280)
            h, w = img.shape[:2]
            max_display_w = 1280
            if w > max_display_w:
                scale = max_display_w / w
                new_w, new_h = int(w * scale), int(h * scale)
                self.preview_image = cv2.resize(img, (new_w, new_h))
            else:
                self.preview_image = img.copy()
            
            # Initial display
            self.display_image(self.preview_image, self.lbl_orig_img)
            self.update_processing()
            self.btn_save.setEnabled(True)

    def update_processing(self):
        if self.preview_image is None:
            return
        
        params = self.get_current_params()
        # Process preview image
        processed = ImageProcessor.process_pipeline(self.preview_image, params)
        self.processed_image = processed
        self.display_image(processed, self.lbl_proc_img)

    def display_image(self, img, label_widget):
        if img is None: return

        # OpenCV BGR -> Qt RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        # Get current Label size
        label_size = label_widget.size()
        
        # Default size fallback
        if label_size.width() < 10 or label_size.height() < 10:
            label_size = self.size() 

        # Scale image to fit Label (maintain aspect ratio)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label_widget.setPixmap(scaled_pixmap)

    def save_image(self):
        if self.original_image is None:
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "enhanced_result.jpg", 
                                                   "Image Files (*.jpg *.png)", options=options)
        if file_path:
            # Process FULL resolution image
            print("Processing full resolution image, please wait...")
            params = self.get_current_params()
            final_result = ImageProcessor.process_pipeline(self.original_image, params)
            
            # Save (supports non-ASCII/Unicode paths)
            ext = os.path.splitext(file_path)[1]
            if not ext: ext = ".jpg"
            success, encoded_img = cv2.imencode(ext, final_result)
            if success:
                encoded_img.tofile(file_path)
                print(f"Saved successfully: {file_path}")
            else:
                print("Failed to save.")

    def resizeEvent(self, event):
        # Update image scaling when window is resized
        if self.original_image is not None:
            self.display_image(self.preview_image, self.lbl_orig_img)
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.lbl_proc_img)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())