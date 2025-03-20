import os


def create_frame_pairs(folder_path):
    frame_pairs = {}
    frame_list = sorted(os.listdir(folder_path))  # List of all frame filenames
    for filename in frame_list:
        if filename.endswith('.jpg'):
            frame_number = int(filename[6:-4])  # Extract frame number
            current_frame = os.path.join(folder_path, filename)

            # Previous frame
            if frame_number > 0:
                prev_frame = os.path.join(folder_path, f"frame_{frame_number-1:04d}.jpg")
                frame_pairs[current_frame] = frame_pairs.get(current_frame, {})
                frame_pairs[current_frame]["previous_frame"] = prev_frame

            # Next frame
            next_frame = os.path.join(folder_path, f"frame_{frame_number+1:04d}.jpg")
            if os.path.exists(next_frame):
                frame_pairs[current_frame] = frame_pairs.get(current_frame, {})
                frame_pairs[current_frame]["next_frame"] = next_frame

    return frame_pairs

def create_video_dict(parent_folder):
    video_dict = {}
    for video_folder in os.listdir(parent_folder):
        video_folder_path = os.path.join(parent_folder, video_folder)
        # Ensure it's a directory
        if os.path.isdir(video_folder_path):
            # Create frame pairs for each video folder
            frame_pairs = create_frame_pairs(video_folder_path)
            # Update video_dict with pairs from this video folder
            video_dict.update(frame_pairs)
    return video_dict

