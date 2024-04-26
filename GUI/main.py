import tkinter as tk
import cv2
from PIL import Image, ImageTk
import datetime
import os


class GUI:
    def __init__(self, master, width, height):
        self.master = master
        master.title("GUI")

        self.current_letter = None
        self.record = False
        self.last_button = None

        self.height = height
        self.width = width

        master.geometry(f"{self.width}x{self.height}")

        self.width_panel1 = int(self.width * 0.1)
        self.width_panel2 = int(self.width * 0.7)
        self.width_panel3 = int(self.width * 0.2)

        self.frame_width = int(self.width_panel2 * 0.8)
        self.frame_height = int(self.height * 0.8)

        self.create_panels()

        self.video_source = cv2.VideoCapture(0)

        self.out = None

        self.update()

    def create_panels(self):
        self.panel1 = tk.Frame(self.master, bg="red", width=self.width_panel1)
        self.panel1.pack(side="left", fill="y")

        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, letter in enumerate(alphabet):
            if letter.lower() in ['j', 'z']:
                button = tk.Button(self.panel1, text=letter, font=("Helvetica", 8), width=5, bg="red")
            else:
                button = tk.Button(self.panel1, text=letter, font=("Helvetica", 8), width=5)
                button.bind("<Button-1>", lambda event, b=button, l=letter: self.on_button_click(b, l))
            button.grid(row=i, column=0, sticky="ew", padx=2, pady=1)

        self.panel2 = tk.Frame(self.master, bg="green", width=self.width_panel2)
        self.panel2.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(self.panel2, bg="white", width=self.frame_width, height=self.frame_height)
        self.canvas.pack(expand=True, padx=50, pady=50)

        self.panel3 = tk.Frame(self.master, bg="blue", width=self.width_panel3)
        self.panel3.pack(side="left", fill="y")

        self.record_button = tk.Button(self.panel3, text="Record", command=self.toggle_record)
        self.record_button.pack(side="top", padx=10, pady=10)

    def on_button_click(self, button, letter):
        if self.last_button:
            self.last_button.config(bg="lightgray")

        button.config(bg="green")

        self.last_button = button

        self.current_letter = letter
        print("Current letter set to:", self.current_letter)

    def toggle_record(self):
        if self.current_letter is None:
            raise ValueError("Nie wybrano litery. Wybierz literę przed rozpoczęciem nagrywania.")

        self.record = not self.record
        if self.record:
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            folder_name = "Data"
            letter_folder = os.path.join(folder_name, self.current_letter)
            file_path = f"{letter_folder}/{current_time}.avi"

            os.makedirs(letter_folder, exist_ok=True)

            self.out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            self.record_button.config(text="Stop Recording")
        else:
            if self.out is not None:
                self.out.release()
                self.record_button.config(text="Record")

    def update(self):
        ret, frame = self.video_source.read()

        if ret:
            #Saved frame in orginal size
            frame = cv2.flip(frame, 1)
            if self.record:
                self.out.write(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo

        self.master.after(10, self.update)

    def __del__(self):
        if self.out is not None:
            self.out.release()

        if self.video_source is not None:
            self.video_source.release()
        cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    width = 1280
    height = 720
    app = GUI(root, width, height)
    root.mainloop()


if __name__ == "__main__":
    main()