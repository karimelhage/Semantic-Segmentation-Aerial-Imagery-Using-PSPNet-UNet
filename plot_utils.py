import matplotlib.pyplot as plt
import cv2
from PIL import Image #For Image Processing
import torch
import torchvision.transforms as transforms
def plot_losses(total_train_losses, total_val_losses, model_name='PSPNet'):
    '''
    Function to plot the losses found during model training

    :param total_train_losses:  list of training losses found in training
    :param total_val_losses: list of validation losses found in training
    :param model_name: name of the model used to train
    :return: plot of losses during training
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_train_losses[:120])
    plt.plot(list(range(len(total_train_losses) + 1))[1:120], total_val_losses[:120])
    plt.legend(['train loss', 'val loss', 'val f1'])
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def plot_predictions(img, msk, pred):
    '''
    Function to plot a prediction mask with the relevant true and prediction mask.

    :param img: the image of which a prediction mask was generated
    :param msk: the true mask of the image
    :param pred: the predicted mask of the image using a semantic segmentation model
    :return:
    '''

    f, axarr = plt.subplots(1, 3, figsize=(25, 20))
    axarr[0].imshow(img)
    axarr[0].set_title('Image')
    axarr[1].imshow(msk, cmap='tab20')
    axarr[1].set_title('True Mask')
    axarr[2].imshow(pred, cmap='tab20')
    axarr[2].set_title('Prediction Mask')
    plt.show()

def make_pred(model, sample_image_paths, sample_mask_paths, index):
    '''
    Function to make a generate a prediction mask from an image
    :param model: the model to be used to make a prediction
    :param sample_image_paths: list with the paths to the images
    :param sample_mask_paths: list with the paths of the true mask
    :param index: index of the image/mask with the file to be extracted
    :return: image, true mask, prediction mask
    '''
    model.cpu()
    model.eval()

    transform = transforms.Compose([transforms.Resize([1024, 1024]),
                                    transforms.ToTensor()])

    pred_image = cv2.imread(sample_image_paths[index], 1)
    pred_image = Image.fromarray(pred_image)
    test_image_size = pred_image.size
    img = transform(pred_image)
    img = img.unsqueeze(0)
    output = model(img)
    output = torch.argmax(output, dim=1).squeeze(0)  # taking max prob of class form predictions per pixel in mask
    img = cv2.resize(np.asarray(pred_image, dtype='uint8'), (test_image_size[0], test_image_size[1]),
                     interpolation=cv2.INTER_NEAREST)
    output = cv2.resize(np.asarray(output, dtype='uint8'), (test_image_size[0], test_image_size[1]),
                        interpolation=cv2.INTER_NEAREST)
    mask = Image.open(sample_mask_paths[index])

    return img, mask, output
