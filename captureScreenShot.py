import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import time

# Create a directory to save the screenshots (if it doesn't exist)
output_dir = "/Users/jinghangli/no-brainer/photos/"
os.makedirs(output_dir, exist_ok=True)

# Initialize the video capture object for the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Create a function to capture and save a screenshot
def capture_screenshot():
    ret, frame = cap.read()
    if not ret:
        return

    # Open a file dialog to select the output directory
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    if not output_directory:
        return

    # Open an entry dialog to input the file name
    file_name = filedialog.asksaveasfilename(
        title="Save As",
        initialdir=output_directory,
        defaultextension=".jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
    )

    if not file_name:
        return

    # Save the screenshot with the chosen file name and in the selected directory
    cv2.imwrite(file_name, frame)
    status_label.config(text=f"Screenshot saved as {file_name}")

    # Load and display the saved screenshot in the GUI
    img = Image.open(file_name)
    img = img.resize((300, 225), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    screenshot_label.config(image=img)
    screenshot_label.image = img

# Create a function to update the live video feed
def update_video_feed():
    ret, frame = cap.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        video_label.config(image=img)
        video_label.image = img
    video_label.after(10, update_video_feed)

# Create the main application window
app = tk.Tk()
app.title("Webcam Screenshot Capture")

# Create a label to display the live video feed
video_label = ttk.Label(app)
video_label.pack()

# Create a button to capture a screenshot
capture_button = ttk.Button(app, text="Capture Screenshot", command=capture_screenshot)
capture_button.pack(pady=10)

# Create a label to display the saved screenshot
screenshot_label = ttk.Label(app)
screenshot_label.pack()

# Create a label to display status messages
status_label = ttk.Label(app, text="")
status_label.pack()

# Create a function to close the webcam and exit the application
def exit_application():
    cap.release()
    app.quit()

# Create an exit button to close the application
exit_button = ttk.Button(app, text="Exit", command=exit_application)
exit_button.pack(pady=10)

# Start updating the live video feed
update_video_feed()

app.mainloop()

