import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageTk

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

import torch

# Class yang merepresentasikan aplikasi deskripsi gambar pake AI
class ImageCaptioningAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning AI")
        self.root.geometry("800x800")

        self.model = VisionEncoderDecoderModel.from_pretrained("NourFakih/image-captioning-Vit-GPT2-Flickr8k")
        self.feature_extractor = ViTImageProcessor.from_pretrained("NourFakih/image-captioning-Vit-GPT2-Flickr8k")
        self.tokenizer = AutoTokenizer.from_pretrained("NourFakih/image-captioning-Vit-GPT2-Flickr8k")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.title_label = tk.Label(self.root, text="Image Captioning AI", font=("Segoe Script", 25, "bold"))
        self.title_label.pack(pady=25)

        self.upload_button = ttk.Button(self.root, text="Upload an image", command=self.upload_image, width=22)
        self.upload_button.pack(pady=10)

        self.image_label = tk.Label(self.root, text="(No image yet)")
        self.image_label.pack(pady=100)

        self.caption_label = tk.Label(self.root, text="Generated caption : -", font=("Arial", 13, "bold"), wraplength=500, anchor="w")
        self.caption_label.pack(pady=10)

        self.caption_button = ttk.Button(self.root, text="Generate caption", command=self.generate_caption, width=22)
        self.caption_button.pack(pady=10)

        self.image_path = None
        self.current_image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.show_image()

    def show_image(self):
        if not self.image_path:
            return
        img = Image.open(self.image_path)

        max_width = 300
        max_height = 300
        img.thumbnail((max_width, max_height))

        self.current_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.current_image)
        self.image_label.image = self.current_image
        

    def generate_caption(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        img = Image.open(self.image_path).convert("RGB")

        pixel_values = self.feature_extractor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)

        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if caption:
            self.caption_label.config(text=f"Generated caption : {caption}")

# Program utamanya
root = tk.Tk()
app = ImageCaptioningAIApp(root)
root.mainloop()