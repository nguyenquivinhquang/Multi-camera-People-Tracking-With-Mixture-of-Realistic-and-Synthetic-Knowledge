"""
@TODO:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Load the ResNet-50 model
    model = ResNet50()

    # Load an input image and pre-process it for the model
    image = load_image(image_path)
    image = preprocess_image(image)

    # Pass the input image through the model to get the predicted class
    prediction = model(image)

    # Get the index of the predicted class
    predicted_class = prediction.argmax()

    # Set the model to evaluation mode
    model.eval()

    # Create a hook that stores the output of the final convolutional layer
    final_conv_layer = model._modules.get('layer4')
    def hook(module, input, output):
        hook.output = output
    final_conv_layer.register_forward_hook(hook)

    # Pass the input image through the model again to get the output of the final convolutional layer
    model(image)

    # Calculate the gradient of the predicted class with respect to the output of the final convolutional layer
    prediction[:, predicted_class].backward()

    # Get the gradient of the output of the final convolutional layer with respect to the input image
    grads = image.grad.data

    # Normalize the gradient
    normalized_grads = grads / (torch.sqrt(torch.mean(torch.square(grads))) + 1e-5)

    # Weight the output of the final convolutional layer by the normalized gradient
    weights = normalized_grads.mean(dim=(1, 2, 3))
    cam = (hook.output * weights[:, None, None, :]).sum(dim=3)

    # Resize the CAM to the same size as the input image
    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Heatmap of the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Superimpose the heatmap on the input image
    result = heatmap * 0.3 + image * 0.5

    # Plot the input image, the heatmap, and the superimposed image
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(heatmap)
    plt.subplot(133)
    plt.imshow(result)
    plt.show()



"""
