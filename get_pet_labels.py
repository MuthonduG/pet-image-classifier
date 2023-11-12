#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Victor B.
# DATE CREATED: 17/08/2020 22:15                                 
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir


def clean_filename(filename):
    words_filename = filename.split("_")
    only_words = [word.strip().lower() for word in words_filename if word.isalpha()]
    pet_name = " ".join(only_words)
    
    print("Filename={} Label={}".format(filename, pet_name))
    
    return pet_name
    
    

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    
    filename_list = listdir(image_dir)
    filenames = []
    results_dic = dict()
    
    for index in range(0, len(filename_list), 1):
        filename = filename_list[index]
        print("{:2d} file: {:>25}".format(index + 1, filename))
        filenames.append(filename)
    
    pet_labels = [[filename, clean_filename(filename)] for filename in filenames]
    
    for index in range(0, len(pet_labels), 1):
        filename = pet_labels[index][0]
        pet_label = pet_labels[index][1]
        
        if filename not in results_dic:
            results_dic[filename] = [pet_label]
        else:
            print("** Warning: Key={} already exist with value={}".format(filename, results_dic[filename]))  

    return results_dic
