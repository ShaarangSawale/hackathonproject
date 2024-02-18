import cv2
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.window.configure(bg='#FFD1DC')

        # Check if "captured_image.jpg" exists and delete it
        if os.path.exists("captured_image.jpg"):
            os.remove("captured_image.jpg")
            print("Previous captured image deleted.")

        # Open the default camera (usually the built-in webcam)
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Couldn't open the webcam.")
            return
        
        self.info_icon = ImageTk.PhotoImage(Image.open("info_icon.png"))
        
        # Button for help
        self.help_button = ttk.Button(window, image = self.info_icon, command=self.show_help)
        #self.help_button.pack(side=tk.LEFT, padx=10)
        self.help_button.grid(row=0, column=0,sticky=('W'))

        #img = ImageTk.PhotoImage(Image.open("logo.png"))

        # Create a Label Widget to display the text or Image
        #self.logo_label = ttk.Label(window, image = img)

        # Add a text label above the camera feed
        #self.text_label = ttk.Label(window, text="LooksMaxxer - Find Your Face Shape", font=('Helvetica', 12, 'bold'))
        #self.text_label.pack(side=tk.RIGHT, pady=5)
        #self.logo_label.grid(row=0, column=1,sticky=('N','W'), padx=10, pady=5)

         # Load the logo image
        self.logo_img = ImageTk.PhotoImage(Image.open("logo.png"))

        # Create a Label Widget to display the logo image
        self.logo_label = ttk.Label(window, image=self.logo_img, style='Pink.TLabel')
        self.logo_label.grid(row=0, column=1, pady=5)

        self.subtitle_label = ttk.Label(window, text="make sure your whole face fits in the gray oval for best results", font=('Gulim', 8))
        #self.text_label.pack(pady=5)
        self.subtitle_label.grid(row=1, column=1, padx = 20, pady=5)

        #self.invis_label = ttk.Label(window)
        #self.text_label.pack(pady=5)
        #self.invis_label.grid(row=0, column=2,sticky=('W'), pady=5)
        
        # Create a label to display the camera feed
        self.label = ttk.Label(window)
        #self.label.pack(side=tk.BOTTOM, padx=10, pady=10)
        self.label.grid(row=2,column=0, columnspan=3)
        
        # Load the overlay image
        self.overlay_image = Image.open("overlay_image.png")

        self.cam_img = ImageTk.PhotoImage(Image.open("camera_icon.png"))

        # Button to capture image
        self.capture_button = ttk.Button(window, image=self.cam_img, text="Capture", compound="left", command=self.capture_image)
        #self.capture_button.pack(pady=10)
        self.capture_button.grid(row=3,column=2, pady=10)

        # Bind closing of the Tkinter window to the release of the camera
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # After setting up GUI, start the camera feed
        self.show_camera_feed()

    def show_help(self):
        # Display help information in a new window
        help_window = tk.Toplevel()
        help_window.title("Info")
        help_window.configure(bg='#FFD1DC')

        self.logo_img_2 = ImageTk.PhotoImage(Image.open("logo.png"))

        # Create a Label Widget to display the logo image
        self.logo_label_2 = ttk.Label(help_window, image=self.logo_img_2, style='Pink.TLabel')
        self.logo_label_2.grid(row=0, column=0, pady=5)

        self.info_img = ImageTk.PhotoImage(Image.open("info_card.png"))

        # Create a Label Widget to display the logo image
        self.info_label = ttk.Label(help_window, image=self.info_img, style='Pink.TLabel')
        self.info_label.grid(row=1, column=0, pady=5)

        # Create a Label widget to display the help text
        #help_label = ttk.Label(help_window, text="This is a test", font=('Helvetica', 12))
        #help_label.grid(row=1, column=0, padx=10, pady=10)  # You can adjust row, column, padx, pady as needed


    def show_camera_feed(self):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.flip(frame, 1)

            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a PIL Image
            pil_frame = Image.fromarray(rgb_frame)

            # Resize the overlay image to match the camera feed dimensions
            overlay_resized = self.overlay_image.resize((pil_frame.width, pil_frame.height))

            # Paste the resized overlay image onto the camera feed
            pil_frame.paste(overlay_resized, (0, 0), overlay_resized)

            # Convert the PIL Image to a PhotoImage
            img = ImageTk.PhotoImage(pil_frame)

            # Update the label with the new image
            self.label.img = img
            self.label.configure(image=img)

            # Schedule the show_camera_feed method to be called after 10 milliseconds
            self.label.after(10, self.show_camera_feed)

    def capture_image(self):
        ret, frame = self.cap.read()

        if ret:
            # Save the captured image
            cv2.imwrite("captured_image.jpg", frame)
            print("Image captured and saved.")

            # Hide the current window
            self.window.withdraw()

            result = ""
            # Open a new window with a heading and text
            self.show_result_window(result)

    def on_close(self):
        # Release the camera and close the Tkinter window
        self.cap.release()
        self.window.destroy()

    def show_result_window(self, result):
        result_window = tk.Toplevel()
        result_window.title("Face Shape Measured")

        self.logo_img_2 = ImageTk.PhotoImage(Image.open("square_result.png"))

        # Create a Label Widget to display the logo image
        self.logo_label_2 = ttk.Label(result_window, image=self.logo_img_2, style='Pink.TLabel')
        self.logo_label_2.grid(row=0, column=0, pady=5)

    def close_result_window(self):
        # Close the result window
        self.window.deiconify()

def main():
    root = tk.Tk()
    app = CameraApp(root, "Camera App")
    root.mainloop()

if __name__ == "__main__":
    main()
