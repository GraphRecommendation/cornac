import numpy as np
from cornac.models.narre.narre import get_data
from tensorflow import keras


def get_attention(train_set, model, batch_size=64, max_num_review=None):
    item_attention_review_pooling = keras.Model(inputs=[model.model.get_layer('input_item_review').input, model.model.get_layer('input_item_uid_review').input, model.model.get_layer('input_item_number_of_review').input], outputs=[model.model.get_layer('Yi').output, model.model.get_layer('item_attention').output])
    A = np.zeros((train_set.num_items, max_num_review))
    for batch_items in train_set.item_iter(batch_size):
        item_reviews, item_uid_reviews, item_num_reviews = get_data(batch_items, train_set, model.max_text_length, by='item', max_num_review=max_num_review)
        Yi, item_attention = item_attention_review_pooling([item_reviews, item_uid_reviews, item_num_reviews], training=False)
        A[batch_items, :item_attention.shape[1]] = item_attention.numpy().reshape(item_attention.shape[:2])
    return A