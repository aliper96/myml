import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import torch.nn as nn
# from PIL import Image

from detr import  DETR

# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # convert list of dictionaries to a single dictionary
    targets_dict = {}
    for target_list in targets:
        for d in target_list:
            for k, v in d.items():
                if k in targets_dict:
                    targets_dict[k].append(v)
                else:
                    targets_dict[k] = [v]

    return images, targets_dict




def main():
    # define dataset
    train_dataset = CocoDetection(root='coco_data/train2017', annFile='coco_data/annotations/instances_train2017.json',
                                  transform=transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()]))

    # define dataloader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=8)

    # define model
    model = DETR(num_classes=90).to(device)

    # define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # define loss function
    loss_fn = nn.CrossEntropyLoss()

    # train loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in tqdm(train_loader):
            images, targets_dict = batch
            # images = torch.stack([transforms.Resize((624, 624))(item) for item in images])
            images = torch.stack(images).to(device)
            targets_dict = {k: [v.to(device) if isinstance(v, torch.Tensor) else v for v in l] for k, l in targets_dict.items()}

            # forward pass
            outputs = model(images)
            if outputs:
                loss = 0
                for i in range(len(images)):
                    out = {'pred_logits': outputs['pred_logits'][i], 'pred_boxes': outputs['pred_boxes'][i]}
                    tgt_dict = {k: v[i] for k, v in targets_dict.items()}
                    loss += loss_fn(out['pred_logits'].to(device), torch.tensor(tgt_dict['category_id']).to(device).long()) + loss_fn(
                        out['pred_boxes'], torch.tensor(tgt_dict['bbox']).float().to(device))
                total_loss += loss.item()
                num_batches += 1

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update learning rate
                lr_scheduler.step()

        # print epoch loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1} loss: {avg_loss}")

if __name__ == '__main__':
    main()