from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch

class PaperclipDataset(Dataset):
    def __init__(self, video_dict):
        #video_path = list(zip(video_dict.keys(), video_dict.values()))
        #self.paths = video_path
        self.video_dict = video_dict  # Store the video dictionary
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((240, 320)),  # Resize to a consistent size half the original dimensions
        ])

    def __len__(self):
        return len(self.video_dict) - 2

    def __getitem__(self, idx):
        actual_idx = idx + 1  # Skip the first frame
        current_frame = list(self.video_dict.keys())[actual_idx]

        previous_frame = self.video_dict[current_frame].get("previous_frame", None)
        next_frame = self.video_dict[current_frame].get("next_frame", None)

        if previous_frame is None or next_frame is None:
            return self.__getitem__((idx + 1) % len(self.video_dict))
        # Load images
        image_prev = Image.open(previous_frame).convert('RGB')
        image_curr = Image.open(current_frame).convert('RGB')
        image_next = Image.open(next_frame).convert('RGB')

        # Transform images
        image_prev_tensor = self.transform(image_prev)
        image_curr_tensor = self.transform(image_curr)
        image_next_tensor = self.transform(image_next)
        target_frame = image_next_tensor

        #Stack the previous and current frames to form a 6-channel input
        context = torch.cat((image_prev_tensor, image_curr_tensor), dim=0)  # Shape: [6, 480, 640]

        return {'target': target_frame,'context':context}

