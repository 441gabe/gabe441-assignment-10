import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from search import perform_search

class SearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Search Application")

        self.search_label = ttk.Label(root, text="Enter search term:")
        self.search_label.pack(pady=5)

        self.search_entry = ttk.Entry(root, width=50)
        self.search_entry.pack(pady=5)

        self.image_path = tk.StringVar()
        self.image_button = ttk.Button(root, text="Upload Image", command=self.upload_image)
        self.image_button.pack(pady=5)

        self.clear_button = ttk.Button(root, text="Clear Image Path", command=self.clear_image_path)
        self.clear_button.pack(pady=5)

        self.image_label = ttk.Label(root, textvariable=self.image_path)
        self.image_label.pack(pady=5)

        self.pc_label = ttk.Label(root, text="Number of Principal Components:")
        self.pc_label.pack(pady=5)

        self.num_principal_components = tk.IntVar(value=0)
        self.pc_spinbox = ttk.Spinbox(root, from_=0, to=100, textvariable=self.num_principal_components)
        self.pc_spinbox.pack(pady=5)

        self.search_button = ttk.Button(root, text="Search", command=self.search)
        self.search_button.pack(pady=5)

        self.results_label = ttk.Label(root, text="Search Results:")
        self.results_label.pack(pady=5)

        self.results_frame = ttk.Frame(root)
        self.results_frame.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path.set(file_path)

    def clear_image_path(self):
        self.image_path.set("")

    def search(self):
        search_term = self.search_entry.get()
        image_path = self.image_path.get()
        num_pc = self.num_principal_components.get()
        result_image_paths, image_scores = perform_search(search_term, image_path, num_pc)

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        if result_image_paths:
            for image_path, score in zip(result_image_paths, image_scores):
                image = Image.open(image_path)
                image = image.resize((150, 150))
                photo = ImageTk.PhotoImage(image)
                label = None
                if num_pc > 0:
                    label = ttk.Label(self.results_frame, image=photo, text=f"Similarity Score: {score}", compound=tk.BOTTOM)
                else: 
                    label = ttk.Label(self.results_frame, image=photo, text=f"Similarity Score: {score[0:1]}", compound=tk.BOTTOM)
                label.image = photo
                label.pack(side=tk.LEFT, padx=5)
        else:
            self.results_label.config(text="No results found.")

if __name__ == "__main__":
            
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()