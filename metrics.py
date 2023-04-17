import numpy as np
import torch

weights_dice = np.load('/Norm_Metrics/weights_dice.npy') #weights of classes
def accuracy(preds_batch, true_mask_batch, device):  # we simply see how many pixels have the same class label per image
    '''
    Function to calculate the accuracy during tran/val batch
    :param preds_batch: the prediction masks made by a model from a train/val batch
    :param true_mask_batch: the true masks of images in the train/val batch
    :param device: device used ('cpu', 'cuda')
    :return: returns the average accuracy of the batch
    '''

    accuracy_batch = []

    for i in range(len(preds_batch)):
        preds = preds_batch[i].to(device)
        true_mask = true_mask_batch[i].to(device)

        preds = torch.argmax(preds, dim=0)  # same as above

        accuracy_batch.append \
            (torch.sum(preds == true_mask).item() / (256 * 256))  # since images/masks resized to 256 x 256

    return np.mean(accuracy_batch)  # average across batch

def weighted_f1(preds_batch, true_mask_batch, device):
    '''
    Function to calculate the f1 dice score based on the ideas discussed in:
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    The function has been adjusted to consider the weights of the classes
    Function to calculate the weight f1 dice score during tran/val batch
    :param preds_batch: the prediction masks made by a model from a train/val batch
    :param true_mask_batch: the true masks of images in the train/val batch
    :param device: device used ('cpu', 'cuda')
    :return: returns the average f1 dice score of the batch
    '''
    f1_batch = []

    for i in range(len(preds_batch)):
        f1_image = []
        preds = preds_batch[i].to(device)
        true_mask = true_mask_batch[i].to(device)

        preds = torch.argmax(preds,
                             dim=0)  # since for each pixel per image output we are predicting the probabilities that each class
        # exists in a pixel, take the maximum probability so that the image output is in the same form as the mask (mask_width,mask_height)

        for label in range(27):  # 26 class + class 0
            if torch.sum(true_mask == label) != 0:
                area_of_intersect = torch.sum((preds == label) * (
                            true_mask == label))  # we the number of pixels that are the same classes within a specific class
                area_of_img = torch.sum(preds == label)  # we check how many pixels exist of that class in prediction
                area_of_label = torch.sum(true_mask == label)  # same for true mask
                f1 = 2 * area_of_intersect / (area_of_img + area_of_label)  # F1 score for one class
                f1_image.append(f1 * weights_dice[0][label])  # f1score weighted based on training labels class weights

        f1_batch.append(np.sum([tensor.cpu() for tensor in f1_image]))
    return np.mean(f1_batch)  # F1 average taken accross batch