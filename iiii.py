import streamlit as st
from ultralytics import YOLO
import cv2
import torch
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
import numpy as np
import tempfile
import time
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, roc_curve
import matplotlib.pyplot as plt
import os
from datetime import datetime


st.title("DeepVision: Real-time Abnormal Activity Detection 👁️")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create directories to save outputs
    output_dir = "output"
    anomaly_frames_dir = os.path.join(output_dir, "anomaly_frames")
    processed_video_dir = os.path.join(output_dir, "processed_videos")
    os.makedirs(anomaly_frames_dir, exist_ok=True)
    os.makedirs(processed_video_dir, exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_video_path = os.path.join(processed_video_dir, f"processed_{timestamp}.mp4")
    
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    st.write("Processing...")
    
    # Load YOLOv8 models
    model_yolo = YOLO("yolov8n.pt")       # Pre-trained YOLO model for person detection and tracking
    model_best = YOLO("best (6).pt")      # Custom YOLO model for anomaly detection
    
    # Set up video capture
    cap = cv2.VideoCapture(tfile.name)
    assert cap.isOpened(), "Error opening video file"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize VideoWriter to save processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as needed
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize tracking for YOLO model
    track_positions_yolo = {}
    track_trajectories_yolo = defaultdict(list)
    track_sizes_yolo = defaultdict(list)  # New tracking for bounding box sizes
    
    # Parameters for size-based anomaly detection
    LARGE_BOX_AREA_THRESHOLD = 0.1 * (frame_width * frame_height)  # For example, 10% of frame area

    
    # Colors for YOLO detections
    COLOR_WRONG_YOLO = (0, 0, 255)
    
    # Thresholds for running detection
    RUNNING_THRESHOLD = 25  # pixels per frame
    
    # Metrics for YOLO
    y_true_yolo = []
    y_pred_yolo = []
    anomaly_scores_yolo = []
    frame_numbers_yolo = []
    anomaly_types_yolo = []
    
    # Metrics for Best model
    y_true_best = []
    y_pred_best = []
    anomaly_scores_best = []
    frame_numbers_best = []
    anomaly_types_best = []
    
    # Placeholder for video display and metrics
    frame_placeholder = st.empty()
    chart_placeholder_yolo = st.empty()
    chart_placeholder_best = st.empty()
    anomaly_text_yolo = st.empty()
    anomaly_text_best = st.empty()
    
    df_metrics_yolo = pd.DataFrame(columns=["Frame", "Anomaly Score", "Anomaly Type"])
    df_metrics_best = pd.DataFrame(columns=["Frame", "Anomaly Score", "Anomaly Type"])
    
    frame_count = 0
    abnormal_count_yolo = 0
    abnormal_count_best = 0
    
    # Define class names for Best model
    class_names_best = {
    0: 'Accident',
    1: 'Fire',
    2: 'Violence',
    3: 'dancing',
    4: 'jumping',
    5: 'throwing',
    6: 'unusual_movement'
    }
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        frame_count += 1

        # YOLO Model Predictions (Person Detection and Tracking)
        results_yolo = model_yolo.track(img.copy(), persist=True, classes=[0], conf=0.7, show=False)

        # Best Model Predictions (Anomaly Detection)
        results_best = model_best(img.copy(), show=False, conf=0.5)
        # print(results_best)
        frame_anomalies_yolo = []
        current_anomaly_score_yolo = 0

        frame_anomalies_best = []
        current_anomaly_score_best = 0

        # Process YOLO results for Person Detection and Tracking
        for track in results_yolo:
            for box in track.boxes:
                if box.id is None:
                    continue  # Skip if track ID is not available
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                track_id = int(box.id[0].item())

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                current_centroid = (centroid_x, centroid_y)
                
                # Calculate the area of the bounding box
                current_size = (x2 - x1) * (y2 - y1)
                
                # Check if bounding box is in the right half of the frame and large enough
                is_wrong_direction = False
                if centroid_x > frame_width // 2 and current_size > LARGE_BOX_AREA_THRESHOLD:
                    is_wrong_direction = True

                anomaly = False
                anomaly_type = ""

                if is_wrong_direction:
                    anomaly = True
                    anomaly_type = "Wrong Direction"
                    current_anomaly_score_yolo += 1

                if anomaly:
                    abnormal_count_yolo += 1
                    y_pred_yolo.append(1)
                    anomaly_types_yolo.append(anomaly_type)
                    frame_anomalies_yolo.append(anomaly_type)
                    color = COLOR_WRONG_YOLO
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_text = f": {anomaly_type} ID: {track_id}"
                    cv2.putText(img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    y_pred_yolo.append(0)
                    anomaly_types_yolo.append("Normal")

                y_true_yolo.append(0)  # Placeholder, replace with actual ground truth

        # Process Best model results for Anomaly Detection
        for det in results_best:
            for box in det.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                current_centroid = (centroid_x, centroid_y)

                anomaly_type = class_names_best.get(cls_id, "Unknown")
                if anomaly_type == "Unknown":
                    continue  # Skip unknown classes

                # Since this model detects specific anomalies, all detections are considered anomalies
                anomaly = True

                if anomaly:
                    abnormal_count_best += 1
                    y_pred_best.append(1)
                    anomaly_types_best.append(anomaly_type)
                    frame_anomalies_best.append(anomaly_type)
                    color = (0, 0, 255)  # Red color for all detections

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{anomaly_type}"
                    cv2.putText(img, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    y_pred_best.append(0)
                    anomaly_types_best.append("Normal")

                y_true_best.append(0)  # Placeholder, replace with actual ground truth

        anomaly_scores_yolo.append(current_anomaly_score_yolo)
        frame_numbers_yolo.append(frame_count)

        anomaly_scores_best.append(current_anomaly_score_best)
        frame_numbers_best.append(frame_count)

        # Update metrics dataframe for YOLO
        new_row_yolo = {
            "Frame": frame_count,
            "Anomaly Score": current_anomaly_score_yolo,
            "Anomaly Type": ", ".join(frame_anomalies_yolo) if frame_anomalies_yolo else "Normal"
        }
        df_metrics_yolo = pd.concat([df_metrics_yolo, pd.DataFrame([new_row_yolo])], ignore_index=True)

        # Update metrics dataframe for Best model
        new_row_best = {
            "Frame": frame_count,
            "Anomaly Score": current_anomaly_score_best,
            "Anomaly Type": ", ".join(frame_anomalies_best) if frame_anomalies_best else "Normal"
        }
        df_metrics_best = pd.concat([df_metrics_best, pd.DataFrame([new_row_best])], ignore_index=True)

        # Save frames with anomalies
        if frame_anomalies_yolo or frame_anomalies_best:
            anomaly_frame_path = os.path.join(anomaly_frames_dir, f"anomaly_frame_{frame_count}.jpg")
            cv2.imwrite(anomaly_frame_path, img)

        # Write the processed frame to the video
        out.write(img)

        # Update charts
        chart_placeholder_yolo.line_chart(df_metrics_yolo.set_index("Frame")["Anomaly Score"], width=0, height=200, use_container_width=True)
        chart_placeholder_best.line_chart(df_metrics_best.set_index("Frame")["Anomaly Score"], width=0, height=200, use_container_width=True)

        # Display anomaly types
        if frame_anomalies_yolo:
            anomaly_text_yolo.markdown(f"**YOLO - Frame {frame_count}:** " + ", ".join(frame_anomalies_yolo))
        else:
            anomaly_text_yolo.markdown(f"**YOLO - Frame {frame_count}:** Normal")

        if frame_anomalies_best:
            anomaly_text_best.markdown(f"**Best - Frame {frame_count}:** " + ", ".join(frame_anomalies_best))
        else:
            anomaly_text_best.markdown(f"**Best - Frame {frame_count}:** Normal")

        # Display frame in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB")

        time.sleep(1 / fps)

    cap.release()
    out.release()

    st.write("Processing Completed.")

    # Provide download links for the processed video and anomaly frames
    with st.expander("Download Outputs"):
        st.markdown(f"**Processed Video:** [Download]({processed_video_path})")
        
        # List anomaly frames
        anomaly_files = os.listdir(anomaly_frames_dir)
        if anomaly_files:
            st.markdown("**Anomaly Frames:**")
            for file in anomaly_files:
                file_path = os.path.join(anomaly_frames_dir, file)
                # Display image
                try:
                    st.image(file_path, caption=file, use_column_width=True)
                except UnidentifiedImageError:
                    st.write(f"Error: Unable to display {file}. File might not be a valid image.")
                # Provide download link
                st.markdown(f"[Download {file}](./{file_path})")
        else:
            st.write("No anomaly frames detected.")
    
    # Metrics Calculation for YOLO
    if len(y_true_yolo) > 0:
        accuracy_yolo = accuracy_score(y_true_yolo, y_pred_yolo)
        try:
            auc_yolo = roc_auc_score(y_true_yolo, y_pred_yolo)
        except:
            auc_yolo = "Undefined"
        rmse_yolo = np.sqrt(mean_squared_error(y_true_yolo, y_pred_yolo))

        st.write("### YOLO Model Metrics")
        st.write(f"**Accuracy:** {accuracy_yolo:.2f}")
        st.write(f"**RMSE:** {rmse_yolo:.2f}")
        st.write(f"**AUC-ROC:** ")
    
        # Plot AUC-ROC if possible for YOLO
        if auc_yolo != "Undefined":
            fpr_yolo, tpr_yolo, _ = roc_curve(y_true_yolo, y_pred_yolo)
            plt.figure()
            plt.plot(fpr_yolo, tpr_yolo, label='YOLO AUC-ROC')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('YOLO AUC-ROC Curve')
            plt.legend()
            st.pyplot(plt)

        # Create a dataframe for YOLO bar charts
        metrics_df_yolo = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC-ROC', 'RMSE'],
            'Value': [accuracy_yolo, auc_yolo if auc_yolo != "Undefined" else 0, rmse_yolo]
        })

        # Plot bar chart for YOLO
        fig_yolo, ax_yolo = plt.subplots()
        bars_yolo = ax_yolo.bar(metrics_df_yolo['Metric'], metrics_df_yolo['Value'], color=['blue', 'green', 'red'])
        ax_yolo.set_ylim(0, max(metrics_df_yolo['Value']) * 1.2)
        for bar in bars_yolo:
            height = bar.get_height()
            ax_yolo.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')
        st.pyplot(fig_yolo)

    # Metrics Calculation for Best model
    if len(y_true_best) > 0:
        accuracy_best = accuracy_score(y_true_best, y_pred_best)
        try:
            auc_best = roc_auc_score(y_true_best, y_pred_best)
        except:
            auc_best = "Undefined"
        rmse_best = np.sqrt(mean_squared_error(y_true_best, y_pred_best))

        st.write("### Best Model Metrics")
        st.write(f"**Accuracy:** {accuracy_best:.2f}")
        st.write(f"**RMSE:** {rmse_best:.2f}")
        st.write(f"**AUC-ROC:**")

        # Plot AUC-ROC if possible for Best model
        if auc_best != "Undefined":
            fpr_best, tpr_best, _ = roc_curve(y_true_best, y_pred_best)
            plt.figure()
            plt.plot(fpr_best, tpr_best, label='Best AUC-ROC')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Best Model AUC-ROC Curve')
            plt.legend()
            st.pyplot(plt)

        # Create a dataframe for Best model bar charts
        metrics_df_best = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC-ROC', 'RMSE'],
            'Value': [accuracy_best, auc_best if auc_best != "Undefined" else 0, rmse_best]
        })

        # Plot bar chart for Best model
        fig_best, ax_best = plt.subplots()
        bars_best = ax_best.bar(metrics_df_best['Metric'], metrics_df_best['Value'], color=['purple', 'orange', 'cyan'])
        ax_best.set_ylim(0, max(metrics_df_best['Value']) * 1.2)
        for bar in bars_best:
            height = bar.get_height()
            ax_best.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')
        st.pyplot(fig_best)

    st.write("Ground Truth Results for YOLO:")
    st.write(y_true_yolo)

    st.write("Ground Truth Results for Best Model:")
    st.write(y_true_best)
