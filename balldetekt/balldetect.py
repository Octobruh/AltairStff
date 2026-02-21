'''
python3 balldetect.py --mode camera
python3 balldetect.py --mode camera --cal 20000
python3 balldetect.py --mode video --input test.mp4 --output final_render.mp4 --cal 12500
'''

import cv2
import math
import argparse
import supervision as sv
from rfdetr import RFDETRBase

def process_frame(frame, model, cal_const, box_annotator, label_annotator):
    """Handles color conversion, prediction, distance math, and drawing for a single frame."""
    
    # Convert BGR (OpenCV) to RGB explicitly
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pass the safe RGB frame to the model
    detections = model.predict(rgb_frame, threshold=0.5)

    custom_labels = []
    center_points = []
    
    # Iterate through detections to calculate the diagonal, distance, and centers
    for i in range(len(detections)):
        x_min, y_min, x_max, y_max = detections.xyxy[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]

        # Calculate bounding box width and height
        w = x_max - x_min
        h = y_max - y_min
        
        # Center point from bounding box
        center_x = int(x_min + w / 2)
        center_y = int(y_min + h / 2)
        center_points.append((center_x, center_y))
        
        # Pythagorean theorem for the diagonal
        diagonal = math.sqrt(w**2 + h**2)

        # Estimate the distance (prevent division by zero)
        distance = cal_const / diagonal if diagonal > 0 else 0.0
        
        # Format the label
        custom_labels.append(f"Class {class_id} {confidence:.2f} | Dist: {distance:.2f}")

    # Draw the bounding boxes and custom labels using Supervision
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, 
        detections=detections, 
        labels=custom_labels
    )
    
    # Draw the solid red center points on top of everything
    for pt in center_points:
        # cv2.circle parameters: image, center, radius, color (BGR), thickness (-1 fills the circle)
        cv2.circle(annotated_frame, pt, radius=5, color=(0, 0, 255), thickness=-1)
    
    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description="RF-DETR Object Detection with Distance Estimation")
    parser.add_argument("--mode", type=str, choices=["camera", "video"], required=True, 
                        help="Choose the run mode: 'camera' for live webcam, 'video' for file processing.")
    parser.add_argument("--input", type=str, 
                        help="Path to the input video file (Required if mode is 'video').")
    parser.add_argument("--output", type=str, default="output_video.mp4", 
                        help="Path to save the processed video (Video mode only).")
    parser.add_argument("--weights", type=str, default="checkpoint_best_total.pth", 
                        help="Path to your custom RF-DETR weights.")
    parser.add_argument("--cal", type=float, default=10000.0, 
                        help="Calibration constant (C) for distance estimation.")
    
    args = parser.parse_args()

    # Model init
    print(f"Loading RF-DETR model with weights: {args.weights}...")
    try:
        model = RFDETRBase(pretrain_weights=args.weights)
    except Exception as e:
        print(f"Could not load custom weights '{args.weights}', falling back to base model. Error: {e}")
        model = RFDETRBase()
        
    model.optimize_for_inference()
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # If using camera
    if args.mode == "camera":
        print("Starting camera stream... Press 'q' in the video window to exit.")
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame from camera. Check your USB connection.")
                break
            
            # Process the frame using our shared function
            annotated_frame = process_frame(frame, model, args.cal, box_annotator, label_annotator)
            
            cv2.imshow("RF-DETR Real-Time Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    # If reading video file
    elif args.mode == "video":
        if not args.input:
            print("Error: You must provide an --input file path when running in video mode.")
            return
            
        video_info = sv.VideoInfo.from_video_path(video_path=args.input)
        print(f"Processing video: {video_info.width}x{video_info.height} at {video_info.fps} FPS")

        with sv.VideoSink(target_path=args.output, video_info=video_info) as sink:
            for frame in sv.get_video_frames_generator(source_path=args.input):
                # Process the frame using our shared function
                annotated_frame = process_frame(frame, model, args.cal, box_annotator, label_annotator)
                sink.write_frame(frame=annotated_frame)
                
                # (Optional) Display the processing in real-time
                cv2.imshow("Processing...", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        print(f"Done! Saved processed video to {args.output}")

if __name__ == "__main__":
    main()

'''
1. List devices in Powershell
usbipd list

2. Bind the device to make it available for sharing (replace <BUSID> with actual ID):
usbipd bind --busid <BUSID>

3. Attach the device to active WSL instance:
usbipd attach --wsl --busid <BUSID>
---

See the device in WSL
v4l2-ctl --list-devices
'''