import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torchvision.transforms import transforms
from torchvision.models import resnet50, ResNet50_Weights

transformations = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = resnet50()
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)

print("Loading model...")
model.load_state_dict(torch.load("model.pth"))
print("Model loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


def is_image_attacked(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        print(output)

        _, predicted = torch.max(output, 1)
        print(predicted)

        return True if predicted else False

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Attack Checker")

        self.image_name_label = tk.Label(self.root, text="", font=("Helvetica", 14))
        self.image_name_label.pack(side="top", fill="both", expand="yes")
        
        # Create a panel to display the image
        self.panel = tk.Label(self.root)
        self.panel.pack(side="top", fill="both", expand="yes")

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side="bottom", fill="both", expand="yes")
        
        # Create a button to load the image
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image, font=("Helvetica", 14))
        self.load_button.pack(side="left", fill="both", expand="yes", padx=10, pady=10)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear_content, font=("Helvetica", 14))
        self.clear_button.pack(side="right", fill="both", expand="yes", padx=10, pady=10)
        
        # Create a text box to show if the image is attacked or not
        self.result_text = tk.Text(self.root, height=1, width=50, font=("Helvetica", 14))
        self.result_text.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)
        self.result_text.tag_configure("attacked", foreground="red")
        self.result_text.tag_configure("not_attacked", foreground="green")
    
    def load_image(self):
        # Open a file dialog to select an image
        image_path = filedialog.askopenfilename()
        
        if image_path:
            try:
                image_name = image_path.split('/')[-1]
                self.image_name_label.config(text=image_name)
                # Open the image using PIL
                img = Image.open(image_path)
                
                # Resize the image to fit the panel
                img = img.resize((100, 100), Image.Resampling.LANCZOS)
                
                # Convert the image to PhotoImage format
                img = ImageTk.PhotoImage(img)
                
                # Update the panel to display the image
                self.panel.configure(image=img)
                self.panel.image = img
                
                # Check if the image is attacked using the dummy model
                attacked = is_image_attacked(image_path)
                
                # Update the text box with the result
                self.result_text.delete(1.0, tk.END)
                if attacked:
                    result = "Image is attacked!"
                    self.result_text.insert(tk.END, result, "attacked")
                else:
                    result = "Image is not attacked!"
                    self.result_text.insert(tk.END, result, "not_attacked")
            
            except Exception as e:
                messagebox.showerror("Error", f"Unable to load image: {e}")

    def clear_content(self):
        # Clear the image name label
        self.image_name_label.config(text="")
        
        # Clear the image from the panel
        self.panel.configure(image='')
        self.panel.image = None
        
        # Clear the text from the text box
        self.result_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
