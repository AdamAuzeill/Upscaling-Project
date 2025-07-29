import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageOps
import subprocess
import os

class ImageCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Image Compare Pro 🔍")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)

        # Two separate image buffers
        self.images = {
            'A': {'path': None, 'original': None, 'current': None, 'zoom': 1.0, 'offset': [0, 0]},
            'B': {'path': None, 'original': None, 'current': None, 'zoom': 1.0, 'offset': [0, 0]},
        }

        self.active_image = 'A'
        self.mode_var = tk.StringVar(value='SLIDER')
        self.auto_open_var = tk.BooleanVar(value=False)
        self.first_image = True

        self.setup_ui()
        self.bind_events()
        
    def setup_ui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", pady=4)

        tk.Label(control_frame, text="Comparison Mode:").pack(side="left", padx=(5, 0))
        for mode in ("SLIDER", "IMAGE A", "IMAGE B"):
            tk.Radiobutton(control_frame, text=mode, variable=self.mode_var,
                           value=mode, command=self.render).pack(side="left")

        tk.Button(control_frame, text="Zoom +", command=lambda: self.adjust_zoom(1.2)).pack(side="right", padx=4)
        tk.Button(control_frame, text="Zoom -", command=lambda: self.adjust_zoom(0.8)).pack(side="right", padx=4)

        for key in ['A', 'B']:
            frame = tk.LabelFrame(self.root, text=f"Image {key} Controls")
            frame.pack(fill="x", pady=2, padx=5)

            tk.Button(frame, text="📂 Open", command=lambda k=key: self.open_image(k)).pack(side="left", padx=5)
            tk.Label(frame, text="Scale:").pack(side="left")
            setattr(self, f"scale_{key}", tk.StringVar(value="2"))
            tk.Entry(frame, textvariable=getattr(self, f"scale_{key}"), width=4).pack(side="left")

            tk.Label(frame, text="Filter:").pack(side="left")
            setattr(self, f"filter_{key}", tk.StringVar(value="NONE"))
            ttk.Combobox(frame, textvariable=getattr(self, f"filter_{key}"),
                         values=["NONE", "SHARPEN", "BLUR", "EDGE"],
                         state="readonly", width=10).pack(side="left", padx=4)

            tk.Label(frame, text="Resample:").pack(side="left")
            setattr(self, f"resample_{key}", tk.StringVar(value="LANCZOS"))
            ttk.Combobox(frame, textvariable=getattr(self, f"resample_{key}"),
                         values=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"],
                         state="readonly", width=10).pack(side="left", padx=4)

            tk.Button(frame, text="Apply Filter", command=lambda k=key: self.apply_filter(k)).pack(side="left", padx=6)
            tk.Button(frame, text="Upscale", command=lambda k=key: self.upscale_image(k)).pack(side="left", padx=6)

        self.canvas = tk.Canvas(self.root, bg="#222")
        self.canvas.pack(fill="both", expand=True)

        self.slider = tk.Scale(self.root, from_=0, to=100, orient="horizontal", command=lambda v: self.render(), length=600)
        self.slider.set(50)
        self.slider.pack(fill="x", padx=50, pady=4)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", pady=4)
        
    
    def bind_events(self):
        self.root.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_wheel)
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
    
    def open_image(self, key):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        
        if path:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img) # corrige l'orientation
            if self.first_image:
                self.images['A'].update({
                    'path': path,
                    'original': img,
                    'current': img.copy(),
                    'zoom': 1.0,
                    'offset': [0, 0]
                })
                self.images['B'].update({
                    'path': path,
                    'original': img,
                    'current': img.copy(),
                    'zoom': 1.0,
                    'offset': [0, 0]
                })
                self.render()
                self.first_image = False
            else:
                self.images[key].update({
                    'path': path,
                    'original': img,
                    'current': img.copy(),
                    'zoom': 1.0,
                    'offset': [0, 0]
                })
                self.render()
    
    def apply_filter(self, key):
        img_data = self.images[key]
        if not img_data['original']:
            return
    
        img = img_data['current'].copy()
        filter_mode = getattr(self, f'filter_{key}').get()
        if filter_mode == "SHARPEN":
            img = img.filter(ImageFilter.SHARPEN)
        elif filter_mode == "BLUR":
            img = img.filter(ImageFilter.BLUR)
        elif filter_mode == "EDGE":
            img = img.filter(ImageFilter.FIND_EDGES)
    
        img_data['current'] = img
        self.render()
    
    
    def upscale_image(self, key):
        img_data = self.images[key]
        if not img_data['original']:
            return
        try:
            scale = max(1, int(getattr(self, f'scale_{key}').get()))
        except:
            return messagebox.showerror("Invalid scale", "Please enter a valid scale.")
    
        resample = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }[getattr(self, f'resample_{key}').get()]
    
        img = img_data['current'].resize((img_data['current'].width * scale, img_data['current'].height * scale), resample=resample)
    
        filter_mode = getattr(self, f'filter_{key}').get()
        if filter_mode == "SHARPEN":
            img = img.filter(ImageFilter.SHARPEN)
        elif filter_mode == "BLUR":
            img = img.filter(ImageFilter.BLUR)
    
        img_data['current'] = img
        self.render()
    
    
    def render(self):
        self.canvas.delete("all")
        a = self.images['A']['current']
        b = self.images['B']['current']
        if not a or not b:
            return
    
        mode = self.mode_var.get()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
    
        # Resize both images keeping their aspect ratios
        za, zb = self.images['A']['zoom'], self.images['B']['zoom']
        a_w, a_h = int(a.width * za), int(a.height * za)
        b_w, b_h = int(b.width * zb), int(b.height * zb)
    
        # Find the maximum width and height for resizing
        target_w = max(a_w, b_w)
        target_h = max(a_h, b_h)
    
        # Maintain the aspect ratio while resizing
        a_img = self.resize_maintain_aspect_ratio(a, target_w, target_h)
        b_img = self.resize_maintain_aspect_ratio(b, target_w, target_h)
    
        offset_x = (self.images['A']['offset'][0] + self.images['B']['offset'][0]) // 2
        offset_y = (self.images['A']['offset'][1] + self.images['B']['offset'][1]) // 2
    
        if mode == "SLIDER":
            composite = Image.new("RGBA", (target_w, target_h))
            composite.paste(b_img)
            split = int(target_w * (self.slider.get() / 100))
            if split > 0:
                composite.paste(a_img.crop((0, 0, split, target_h)), (0, 0))
            tk_img = ImageTk.PhotoImage(composite)
            self.tk_img = tk_img
            self.canvas.create_image(cw//2 + offset_x, ch//2 + offset_y, image=tk_img, anchor="center")
            split_line = cw//2 + offset_x - target_w//2 + split
            self.canvas.create_line(split_line, 0, split_line, ch, fill="red", width=2)
        elif mode == "IMAGE A":
            tk_a = ImageTk.PhotoImage(a_img)
            self.tk_a = tk_a
            self.canvas.create_image(cw//2 + offset_x, ch//2 + offset_y, image=tk_a, anchor="center")
        elif mode == "IMAGE B":
            tk_b = ImageTk.PhotoImage(b_img)
            self.tk_b = tk_b
            self.canvas.create_image(cw//2 + offset_x, ch//2 + offset_y, image=tk_b, anchor="center")

    def resize_maintain_aspect_ratio(self, img, target_w, target_h):
        # Calculate aspect ratio
        aspect_ratio = img.width / img.height
    
        if img.width > img.height:
            new_w = target_w
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = target_h
            new_w = int(new_h * aspect_ratio)
    
        # Resize the image maintaining aspect ratio
        return img.resize((new_w, new_h), Image.LANCZOS)

    
    def adjust_zoom(self, factor):
        self.images['A']['zoom'] *= factor
        self.images['B']['zoom'] *= factor
        self.render()

    def on_ctrl_wheel(self, event):
        if event.state & 0x0004:  # CTRL held
            direction = 1.1 if event.delta > 0 else 0.9
            self.adjust_zoom(direction)

    def start_pan(self, event):
        self.last_mouse_pos = (event.x, event.y)

    def pan_image(self, event):
        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]
        self.images['A']['offset'][0] += dx
        self.images['A']['offset'][1] += dy
        self.images['B']['offset'][0] += dx
        self.images['B']['offset'][1] += dy
        self.last_mouse_pos = (event.x, event.y)
        self.render()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompareApp(root)
    root.mainloop()
