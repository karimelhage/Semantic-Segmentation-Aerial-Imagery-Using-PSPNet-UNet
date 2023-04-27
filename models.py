from numpy import np
from tqdm import tqdm
import torch
from torch.nn.functional import one_hot  # function to one-hot encode labels
import torch.nn as nn
from metrics import weighted_f1, accuracy


def train_smp_model(model, optimizer, epochs, train_dataloader, val_dataloader, scheduler, device='cpu', model_name = 'PSP'):
    '''
    Function to train smp U-Net model

    :param model: The input U-Net model to train
    :param optimizer: torch.optim - the optimizer to be used to train
    :param epochs: int - the number of training epochs
    :param train_dataloader: torch.utils.data.DataLoader - the dataloader with the training images and masks
    :param val_dataloader: torch.utils.data.DataLoader - the dataloader with the validation images and masks
    :param scheduler: torch.optim.lr_scheduler - the learning rate scheduler to decay over training
    :param device: str - the device to be used.
    :return: the trained model along with the associated training and validation losses, accuracies, dice f1 scores
    '''
    criterion = nn.CrossEntropyLoss().to(device)
    total_train_losses = []
    total_train_accuracy = []
    total_train_f1 = []
    total_val_losses = []
    total_val_accuracy = []
    total_val_f1 = []
    for epoch in range(1, epochs + 1):
        ##TRAINING##
        model.train()
        train_losses = []
        train_accuracy = []
        train_f1 = []

        for i, batch, in enumerate(tqdm(train_dataloader)):
            img_batch, lbl_batch = batch
            img_batch = torch.flatten(img_batch, 0,
                                      1)  # since each entru in batch has 6 patches, tensor flattened to instead have 6 patch * batch_size
            lbl_batch = torch.flatten(lbl_batch, 0, 1)  # same as above
            lbl_batch_1hot = one_hot(lbl_batch,
                                     27)  # masks one-hot encoded (only to be used in calculating cross entropy loss between prediction of classes and true class)
            lbl_batch_1hot = lbl_batch_1hot.permute(0, 3, 1,
                                                    2)  # one-hot encoded mask reshaped to (num_masks_in_batch,27,image_width)
            lbl_batch_1hot = lbl_batch_1hot.float()
            img_batch, lbl_batch_1hot = img_batch.to(device), lbl_batch_1hot.to(device)

            optimizer.zero_grad()
            outputs = model(img_batch)

            loss = criterion(outputs, lbl_batch_1hot)
            loss.backward()
            optimizer.step()

            f1 = weighted_f1(outputs, lbl_batch.to(device), device)
            acc = accuracy(outputs, lbl_batch.to(device), device)
            train_losses.append(loss.item())
            train_accuracy.append(acc)
            train_f1.append(f1)

        print(
            f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, f1: {np.mean(train_f1):.3f}, accuracy: {np.mean(train_accuracy):.3f}')
        total_train_losses.append(np.mean(train_losses))
        total_train_accuracy.append(np.mean(train_accuracy))
        total_train_f1.append(np.mean(train_f1))

        ##VALIDATION##
        model.eval()
        val_losses = []
        val_accuracy = []
        val_f1 = []

        for i, batch, in enumerate(tqdm(val_dataloader)):
            img_batch, lbl_batch = batch
            img_batch = torch.flatten(img_batch, 0, 1)
            lbl_batch = torch.flatten(lbl_batch, 0, 1)
            lbl_batch_1hot = one_hot(lbl_batch, 27)
            lbl_batch_1hot = lbl_batch_1hot.permute(0, 3, 1, 2)
            lbl_batch_1hot = lbl_batch_1hot.float()
            img_batch, lbl_batch_1hot = img_batch.to(device), lbl_batch_1hot.to(device)

            with torch.no_grad():
                outputs = model(img_batch)
                loss = criterion(outputs, lbl_batch_1hot)

            f1 = weighted_f1(outputs, lbl_batch.to(device), device)
            acc = accuracy(outputs, lbl_batch.to(device), device)
            val_losses.append(loss.item())
            val_accuracy.append(acc)
            val_f1.append(f1)

        print(
            f'VAL       Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, f1: {np.mean(val_f1):.3f}, accuracy: {np.mean(val_accuracy):.3f}')
        total_val_losses.append(np.mean(val_losses))
        total_val_accuracy.append(np.mean(val_accuracy))
        total_val_f1.append(np.mean(val_f1))
        scheduler.step()

        # Saving the best models if meeting min f1 criteria (this is updated per model trained and epoch)
        if np.mean(val_f1) > min_val_f1:
            torch.save(model.state_dict(),
                       f'/content/drive/MyDrive/FDL/Hurricane Harvey/Models/Unet_pat_aug' + str(epoch) + '.pt')
            min_val_f1 = np.mean(val_f1)

        print(
            f'VAL       Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, f1: {np.mean(val_f1):.3f}, accuracy: {np.mean(val_accuracy):.3f}')
        total_val_losses.append(np.mean(val_losses))
        total_val_accuracy.append(np.mean(val_accuracy))
        total_val_f1.append(np.mean(val_f1))
        scheduler.step()

        # Saving the best model
        if np.mean(val_f1) > min_val_f1:
            torch.save(model.state_dict(),
                       f'Models/{model_name}_pat_mid_' + str(epoch) + '.pt')
            min_val_f1 = np.mean(val_f1)

    return model, total_train_losses, total_train_accuracy, total_train_f1, \
        total_val_losses, total_val_accuracy, total_val_f1
