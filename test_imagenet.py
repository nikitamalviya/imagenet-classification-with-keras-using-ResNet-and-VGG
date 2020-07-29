''' This module tests the accuracy of VGG and ResNet models trained on ImageNet dataset.'''

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
import numpy as np
import argparse
import cv2
import click

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path to the input image")
ap.add_argument("--target_size", required=False, help="image size to reshape")
args = vars(ap.parse_args())

def load_and_preprocess_image(args):
    # load image
    orig = cv2.imread(args["image"])
    # load the input image using the Keras helper utility while ensuring
    click.secho(f"Loading and preprocessing image using Keras utility...", fg='blue')
    image = image_utils.load_img(args["image"], target_size=(224, 224))
    # image = image_utils.load_img(args["image"], target_size=args["target_size"])
    # convert input image into array to get (224, 224, 3)
    image = image_utils.img_to_array(image)
    # to expand the dimensions to be (1, 3, 224, 224)
    image = np.expand_dims(image, axis=0)
    # preprocess image
    preprocessed_image = preprocess_input(image)
    return orig, preprocessed_image

def classify_using_resnet50(image, dataset="imagenet"):
    # load the VGG16 network pre-trained on the ImageNet dataset
    click.secho(f"\nLoading ResNet50 network...", fg='blue')
    click.secho(f"NOTE : It can take upto 5 mins or more to download the ResNet if not present!", fg='magenta')
    model = ResNet50(weights=dataset)
    # classify the image
    click.secho("Predicting class of the image...", fg="green")
    preds = model.predict(image)
    P = decode_predictions(preds)
    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
    return P
    

def classify_using_vgg(image, dataset="imagenet"):
    # load the VGG16 network pre-trained on the ImageNet dataset
    click.secho(f"\nLoading VGG network...", fg='blue')
    click.secho(f"NOTE : It can take upto 5 mins or more to download the VGG if not present!", fg='magenta')
    model = VGG16(weights=dataset)
    # classify the image
    click.secho("Predicting class of the image...", fg="green")
    preds = model.predict(image)
    P = decode_predictions(preds)
    # loop over the predictions and display the rank-5 predictions +
    # probabilities to our terminal
    for (i, (imagenetID, label, prob)) in enumerate(P[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
    return P

def draw_the_predictions_on_image(orig, prediction):
    # load the image via OpenCV, draw the top prediction on the image,
    # and display the image to our screen
    (imagenetID, label, prob) = prediction[0][0]
    cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # cv2.imshow("Classification", orig)
    # cv2.waitKey(0)
    return orig

def main(args):
    # load and process image
    orig, preprocessed_image = load_and_preprocess_image(args)
    
    # use VGG
    # prediction = classify_using_vgg(preprocessed_image, dataset="imagenet")
    
    # use ResNet
    prediction = classify_using_resnet50(preprocessed_image, dataset="imagenet")

    # draw predictions on the image
    classified_image = draw_the_predictions_on_image(orig, prediction)
    # save output
    cv2.imwrite(f"output/{args['image']}", classified_image)

main(args)