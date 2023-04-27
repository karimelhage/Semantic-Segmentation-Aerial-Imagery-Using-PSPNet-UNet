import matplotlib.pyplot as plt


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