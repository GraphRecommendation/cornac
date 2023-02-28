import numpy as np


def get_reviews(eval_method, model, match):
    mnr = model.max_num_review
    reviews = np.zeros((eval_method.train_set.num_items, mnr))
    ui_rid = {}
    for iid, urid in eval_method.review_text.item_review.items():
        for i, (uid, rid) in enumerate(urid.items()):
            if i >= 50:
                break

            ui_rid[(iid, uid)] = rid
            reviews[iid, i] = rid

    indices = model.A.argmax(1)
    best_reviews = np.take_along_axis(reviews, np.expand_dims(indices, axis=1), axis=1)

    return best_reviews
