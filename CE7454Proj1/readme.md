Face parsing assigns pixel-wise labels for each semantic components, e.g., eyes, nose, mouth. The goal of this mini challenge is to design and train a face parsing network. We will use the data from the CelebAMask-HQ Dataset. For this challenge, we prepared a mini-dataset, which consists of 1000 training and 100 validation pairs of images, where both images and annotations have a resolution of 512 x 512.

We follow the setting of CelebAMask-HQ. You can find the definition in color and Document.

You should use your matric number as your displayname.
The final output is NOT an RGB image but a SINGLE-CHANNEL image. Please refer to this code snippet to generate the final output.
If the submission fails, there is a 99.99% chance that it is due to a file structure error. ALL MASKS MUST BE PLACED IN A FOLDER NAMED "masks". IF IT STILL FAILS, PLEASE COMPARE YOUR SUBMISSION WITH THIS SAMPLE.
If your score is very low, please check the format of your mask against THIS SAMPLE.
Updated FAQ for Test Phase

Input Images for Test Phase: You can find them in file page. 

YOU ONLY HAVE A MAXIMUM OF 10 SUBMISSIONS. Failed submissions WILL NOT COUNT TOWARD THIS LIMIT.

DON'T FORGET TO INCLUDE YOUR CODE AND MODDELS IN THE SUBMISSIONS! A sample submission file can be found at this link.

ONLY the FINAL submission will be used as the official result for this competition. This is to prevent cases where some participants achieve better scores using non-compliant models, making it impossible to rectify the issue.

Canceled submissions still count toward the limit due to a CodaBench bug (Confirmed by the developers). They will be periodically deleted during the test phase to restore the limit.

Some submissions may hang for hours due to a CodaBench bug (Confirmed by the developers). Cancel them if no result appears within an hour, then resubmit.

Check the format of your submission:

          submission.zip (correct)

          --solution

          --masks


          submission.zip (error)

          --submission

          ----solution

          ----masks

Restrictions
To maintain fairness, your model should contain fewer than 1,821,085 trainable parameters.
No external data and pretrained models are allowed in this mini challenge. You are only allowed to train your models from scratch using the 1000 image pairs in our given training dataset.
You should not use an ensemble of models.

We use the following code to measure similarity between the predicted and ground truth masks using F-Score.

import numpy as np

from PIL import Image

def compute_multiclass_fscore(mask_gt, mask_pred, beta=1):
    f_scores = []

    for class_id in np.unique(mask_gt):
        tp = np.sum((mask_gt == class_id) & (mask_pred == class_id))
        fp = np.sum((mask_gt != class_id) & (mask_pred == class_id))
        fn = np.sum((mask_gt == class_id) & (mask_pred != class_id))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f_score = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-7)
        )

        f_scores.append(f_score)

    return np.mean(f_scores)
		
		
gt_mask = np.array(Image.open('gt_mask.png').convert("P"))
pred_mask = np.array(Image.open('pred_mask.png').convert("P"))
f_scores=compute_multiclass_fscore(gt_mask, pred_mask)

Submission File Structure
Submissions that do not follow the structure cannot be properly evaluated, which may affect your final marks.

whatever.zip
└── masks
|  └── (Make sure filename is consistent with input, but ends with "png")
|  └── 40159242f6.png
|  └── ...
└── solution
   └── (INCLUDE THIS FOLDER DURING TEST PHASE ONLY)
   └── ckpt.pth          <-- This file MUST exist
   └── requirements.txt  <-- This file MUST exist
   └── run.py            <-- This file MUST exist
A sample submission file can be found at this link.

Please make sure that the code in the solution folder can be invoked with the following command

pip install -r requirements.txt
python3 run.py --input /path/to/input-image.jpg --output /path/to/output-mask.png --weights ckpt.pth
We will use an external program to automatically invoke your code and compare the output-mask.png with the results in the masks folder. Marks will be deducted for incomplete submissions, such as missing key code components, inconsistencies between predictions and code.