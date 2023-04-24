from torchvision.datasets import CocoCaptions, CocoDetection
from torch.utils.data import DataLoader, Dataset


class CocoCaptionsDataset(Dataset):
    def __init__(self, root, annFile, transform=None, tokenizer=None):
        self.coco_captions = CocoCaptions(root=root, annFile=annFile, transform=transform)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.coco_captions)

    def __getitem__(self, idx):
        image, captions = self.coco_captions[idx]

        # Tokenize the first caption as an example
        text = self.tokenizer(captions[0], return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        input_ids = text['input_ids'].squeeze()

        return image, input_ids