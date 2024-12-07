Project Structure
iiii.py: Main Python file to launch the Streamlit application.
requirements.txt: Contains all the dependencies required to run the project.
best (5).pt: Pre-trained model file used for anomaly detection.
yolov8n.pt: Additional model file (if applicable).

Prerequisites
Python 3.9 or higher is recommended.
Ensure pip is installed for package management.
Setup Instructions
1. Install Python and Virtual Environment
Download and install Python from python.org.
Navigate to the project directory in your terminal or command prompt.
2. Create and Activate a Virtual Environment
To keep the project dependencies isolated, it’s recommended to use a virtual environment.


On macOS/Linux
bash
Copy code
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
After activating the virtual environment, install the required dependencies using requirements.txt.

bash
Copy code
pip install -r requirements.txt
4. Running the Project
To start the Streamlit application:

bash
Copy code
streamlit run iiii.py
This command will open the Streamlit interface in your default web browser, where you can interact with the anomaly detection model.