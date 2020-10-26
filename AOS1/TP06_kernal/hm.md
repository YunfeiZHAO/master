In the paper, they create 10 classifier for 0-9 and in their result, digit '9' had the worst result. To simplify our mission, we will create only one classifier for digital '9'. Because The object of this homework is to analyse their method to improve the performance of SVM. So create 9 classifier and combine them to do the classification of 10 digits is not the most important part. For the digit classification method, we can realise this by doing the classification of one digit on these 10 classifiers and if the result is positive for several classifiers, we can compare the  estimated probability to chose which digit it is classified. But this process take long time. For us, training a tradition model and doing the prediction will take 13 minutes  and training a pixel jittered model and doing the prediction will take 30 minutes. As a matter of fact, for  pixel jittered model, we need to do the training process two times on the original dataset and new generated dataset by moving support vectors of the first model respectively  and also with more support vectors, the process for prediction is longer. So we need totally more then 430 minutes.  As we know that the core object to this homework is to analyse the improvement of  model accuracy by incorporating knowledge about invariances of the problem. We decide to take only the classifier of digit '9' as an example.

## Missions

* Do the classification of digit '9' by tradiction SVM with hyperparameters provided by paper
* Implement the  1-pixel jittered model proposed by the paper, and test the performance of this performance on digit '9'
* Compare the performance (accuracy) of these two models
* Explanation of the observation

## Algorithm

1, Train an SVM to generate a set of support vectors $\{S_1,...,S_n\}$  

2,  Generate the artificial examples *(virtual support vectors)* by applying the desired invariance transformations to $\{S_1,...,S_n\}$. In our case, we take the points of support vectors. For each point we generate four new points by moving this point to up one pixel, down one pixel, left one pixel and right one pixel. We take these new points to create a new dataset which is 4 

