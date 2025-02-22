#!/usr/bin/env python

import numpy as np
import random

from utils.utils import sigmoid, softmax, get_negative_samples


def naive_softmax_loss_and_gradient(
        input_vector,
        output_word_idx,
        output_vectors,
        dataset):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between an input word's
    embedding and an output word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    input_vector -- numpy ndarray, input word's embedding
                    in shape (word vector length, )
                    (v_i in the pdf handout)
    output_word_idx -- integer, the index of the output word
                    (o of u_o in the pdf handout)
    output_vectors -- output vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    grad_input_vec -- the gradient with respect to the input word vector
                    in shape (word vector length, )
                    (dL / dv_i in the pdf handout)
    grad_output_vecs -- the gradient with respect to all the output word vectors
                    in shape (num words in vocab, word vector length) 
                    (dL / dU)
    """

    # loss
    scores = np.dot(output_vectors, input_vector)
    exp_scores = np.exp(scores)
    norm_factor = np.sum(exp_scores)
    softmax_probs = exp_scores / norm_factor
    loss = -scores[output_word_idx] + np.log(norm_factor)

    # grad_input_vec, grad_output_vecs
    grad_input_vec = -output_vectors[output_word_idx] + np.dot(softmax_probs, output_vectors)
    grad_output_vecs = np.outer(softmax_probs, input_vector)
    grad_output_vecs[output_word_idx] -= input_vector
  

    return loss, grad_input_vec, grad_output_vecs


def neg_sampling_loss_and_gradient(
        input_vector,
        output_word_idx,
        output_vectors,
        dataset,
        K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a inputWordVec
    and a outputWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an output word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naive_softmax_loss_and_gradient
    """

    # Negative sampling of words is done for you.
    neg_sample_word_indices = get_negative_samples(output_word_idx, dataset, K)
    indices = [output_word_idx] + neg_sample_word_indices

    ### YOUR CODE HERE (~10 Lines)
    ### Please use your implementation of sigmoid in here.
    
    # Initialize loss and gradients
    loss = 0.0
    grad_input_vec = np.zeros(input_vector.shape)
    grad_output_vecs = np.zeros(output_vectors.shape)

    true_output_vec = output_vectors[output_word_idx]
    true_score = np.dot(input_vector, true_output_vec)
    true_prob = sigmoid(true_score)
    loss += -np.log(true_prob)

    # Gradient for true output word
    grad_input_vec += (true_prob - 1) * true_output_vec
    grad_output_vecs[output_word_idx] += (true_prob - 1) * input_vector

    # Negative samples
    for neg_idx in neg_sample_word_indices:
        neg_output_vec = output_vectors[neg_idx]
        neg_score = np.dot(input_vector, neg_output_vec)
        neg_prob = sigmoid(-neg_score)  
        loss += -np.log(neg_prob)

        # Gradient for negative samples
        grad_input_vec += (1 - neg_prob) * neg_output_vec
        grad_output_vecs[neg_idx] += (1 - neg_prob) * input_vector

    ### END YOUR CODE

    return loss, grad_input_vec, grad_output_vecs


def skipgram(current_input_word, window_size, output_words, word2_ind,
             input_vectors, output_vectors, dataset,
             word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_input_word -- a string of the current center word
    window_size -- integer, context window size
    output_words -- list of no more than 2*windowSize strings, the outside words
    word2_ind -- a dictionary that maps words to their indices in
              the word vector list
    input_vectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)
    output_vectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)
    dataset -- dataset for generating negative samples
    word2vec_loss_and_gradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.
    Return:
    loss -- the loss function value for the skip-gram model
            (L in the pdf handout)
    grad_input_vecs -- the gradient with respect to the center word vectors
            (dL / dV, this should have the same shape with V)
    grad_output_vecs -- the gradient with respect to the outside word vectors
                        (dL / dU)
    """

    loss = 0.0
    grad_input_vecs = np.zeros(input_vectors.shape)
    grad_output_vecs = np.zeros(output_vectors.shape)

    # Get the index of the current input word
    input_word_idx = word2_ind[current_input_word]
    input_vector = input_vectors[input_word_idx]

    # Loop over all context words
    for output_word in output_words:
        # Get the index of the output word
        output_word_idx = word2_ind[output_word]

        # Compute loss and gradients for the current context word
        curr_loss, curr_grad_input, curr_grad_output = word2vec_loss_and_gradient(
            input_vector, output_word_idx, output_vectors, dataset
        )

        # Accumulate loss and gradients
        loss += curr_loss
        grad_input_vecs[input_word_idx] += curr_grad_input
        grad_output_vecs += curr_grad_output

    return loss, grad_input_vecs, grad_output_vecs


def word2vec_sgd_wrapper(word2vec_model, word2_ind, word_vectors, dataset,
                         window_size,
                         word2vec_loss_and_gradient=naive_softmax_loss_and_gradient):
    batch_size = 50
    loss = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]

    input_vectors = word_vectors[:int(N / 2), :]
    output_vectors = word_vectors[int(N / 2):, :]

    for i in range(batch_size):
        window_size1 = random.randint(1, window_size)
        input_word, context = dataset.get_random_context(window_size1)

        c, gin, gout = word2vec_model(
            input_word, window_size1, context, word2_ind, input_vectors,
            output_vectors, dataset, word2vec_loss_and_gradient
        )
        loss += c / batch_size
        grad[:int(N / 2), :] += gin / batch_size
        grad[int(N / 2):, :] += gout / batch_size

    return loss, grad
