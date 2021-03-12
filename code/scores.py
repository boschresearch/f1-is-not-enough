""" F1 is Not Enough! Models and Evaluation Towards User-Centered Explainable Question Answering (EMNLP 2020).
Copyright (c) 2021 Robert Bosch GmbH
@author: Hendrik Schuff
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# This file is based on https://github.com/qipeng/golden-retriever/ which in turn is based on https://github.com/hotpotqa/hotpot.

import numpy as np
import json


# The following implementations assume a predictor object that implements a "predict_articles" method.
# This method takes a data set and returns the model's answers, supporting fact predictions, predictions details and
# a list of question ids to which no answer could be predicted.
# You have to implement this method if you want to reuse our score calculation procedure.
# The required details are:
#   * yp1, yp2 and prediction_type as defined in the implementation by Qi et al. and our code in code/select_and_forget.py.
#   * context_token_idx2_fact: A mapping from context token id to a fact.
#   * relevant_fact_probabilities: The estimated relevance probabilities of each fact that is returned as relevant.
#   * irrelevant_facts: Analogous to sp_dict, but with the facts that were predicted to be irrelevant.
#   * irrelevant_fact_probabilities: The estimated relevance probabilities of each fact that is returned as irrelevant.
# We use shelves to speed-up prediction, however, this is not essential to the scores' calculation.

# LocA
def loca_score(predictor,
               shlv,
               data_path='hotpot_dev_distractor_v1.json',
               use_small_test_set=False):
    with open(data_path) as json_file:
        dev_data = json.load(json_file)
    if use_small_test_set:
        print("USING REDUCED DEV SET!")
        dev_data = dev_data[:10]
    dev_data_dict = {}
    for e in dev_data[:]:
        dev_data_dict[e["_id"]] = e
    # Sort consistently
    dev_data = [dev_data_dict[e] for e in sorted(dev_data_dict)]
    # We directly predict a bunch of details that we need for the fact removal as well as the answer in fact measures.
    answer_dict, sp_dict, details_dict, ignored_ids = predictor.predict_articles(dev_data,
                                                                                 details=["yp1",
                                                                                          "yp2",
                                                                                          "prediction_type",
                                                                                          "context_token_idx2_fact"
                                                                                          ],
                                                                                 shlv=shlv)
    if len(ignored_ids):
        raise Exception("Some instances were not predicted successfully!")
    span_type_answer_counter = 0
    span_answer_is_in_proper_fact_and_relevant_counter = 0
    span_answer_is_in_proper_fact_but_irrelevant_counter = 0
    scores = {}
    for _id in answer_dict.keys():
        yp1 = details_dict[_id]['yp1']
        yp2 = details_dict[_id]['yp2']
        prediction_type = details_dict[_id]['prediction_type']
        sps = sp_dict[_id]
        context_token_idx2_fact = details_dict[_id]['context_token_idx2_fact']
        if prediction_type == 'span':
            span_type_answer_counter += 1
            flat_context_token_idx2_fact = [item for sublist in context_token_idx2_fact for item in sublist]
            yp1_fact, yp2_fact = flat_context_token_idx2_fact[yp1], flat_context_token_idx2_fact[yp2]
            # answer spread over multiple facts
            if yp1_fact != yp2_fact:
                pass
            # answer in title
            elif yp1_fact[0] is None:
                pass
            # answer not in title
            else:
                # This asks: Did the model label the fact, in which the answer span lies, as relevant?
                # properly out
                if yp1_fact in [(e[0], e[1]) for e in sps]:
                    span_answer_is_in_proper_fact_and_relevant_counter += 1
                # properly out
                else:
                    span_answer_is_in_proper_fact_but_irrelevant_counter += 1
    scores['A'] = span_type_answer_counter
    scores['I'] = span_answer_is_in_proper_fact_and_relevant_counter
    scores['O'] = span_answer_is_in_proper_fact_but_irrelevant_counter
    # LocA score as defined in Equation 8 in the paper
    scores['LocA'] = scores['I'] / (scores['A'] + scores['O'])
    return scores


# FaRM
def farm_scores(predictor,
                shlv,
                data_path='hotpot_dev_distractor_v1.json',
                use_small_test_set=False,
                num_removals=5):
    with open(data_path) as json_file:
        dev_data = json.load(json_file)
    if use_small_test_set:
        print("USING REDUCED DEV SET!")
        dev_data = dev_data[:10]
    dev_data_dict = {}
    for e in dev_data[:]:
        dev_data_dict[e["_id"]] = e
    # Sort consistently
    dev_data = [dev_data_dict[e] for e in sorted(dev_data_dict)]
    # We directly predict a bunch of details that we need for the fact removal as well as the answer in fact measures.
    answer_dict, sp_dict, details_dict, ignored_ids = predictor.predict_articles(dev_data,
                                                                                 details=["prediction_type",
                                                                                          "relevant_fact_probabilities",
                                                                                          "irrelevant_facts",
                                                                                          "irrelevant_fact_probabilities",
                                                                                          ],
                                                                                 shlv=shlv)
    if len(ignored_ids):
        raise Exception("Some instances where not predicted successfully!")

    scores = {}
    # Relevantly-labelled facts
    new_data_dict_removed_0 = remove_relevantly_labelled_facts_ordered(dev_data_dict,
                                                                       sp_dict,
                                                                       details_dict,
                                                                       num_removed=0)
    # Sort
    new_data_removed_0 = [new_data_dict_removed_0[e] for e in sorted(new_data_dict_removed_0)]

    answer_dict_removed_0, sp_dict_removed_0, details_dict_removed_0, ignored_ids_removed_0 = predictor.predict_articles(
        new_data_removed_0,
        details=["prediction_type",
                 "relevant_fact_probabilities"],
        shlv=shlv)
    assert (new_data_removed_0 == dev_data)
    assert (answer_dict_removed_0 == answer_dict)
    assert (sp_dict_removed_0 == sp_dict)
    new_data_dicts = {0: new_data_removed_0}
    new_answer_dicts = {0: answer_dict_removed_0}
    for k in range(1, num_removals):
        new_data_dicts[k] = remove_relevantly_labelled_facts_ordered(dev_data_dict,
                                                                     sp_dict,
                                                                     details_dict,
                                                                     num_removed=k)
        # Sort
        new_data_dicts[k] = [new_data_dicts[k][e] for e in sorted(new_data_dicts[k])]
        new_answer_dicts[k], _, _, _ = predictor.predict_articles(new_data_dicts[k],
                                                                  details=[
                                                                      "prediction_type"],
                                                                  shlv=shlv)
    changed_answer_fractions_rel = {}
    for k in new_answer_dicts.keys():
        changed_answer_fractions_rel[k] = changed_answer_fraction(new_answer_dicts[0], new_answer_dicts[k])
    scores['c_{rel}(k)'] = changed_answer_fractions_rel

    # Irrelevantly-labelled facts
    new_data_dict_removed_0 = remove_irrelevantly_labelled_facts_ordered(dev_data_dict,
                                                                         sp_dict,
                                                                         details_dict,
                                                                         num_removed=0)
    # Sort
    new_data_removed_0 = [new_data_dict_removed_0[e] for e in sorted(new_data_dict_removed_0)]
    answer_dict_removed_0, sp_dict_removed_0, details_dict_removed_0, ignored_ids_removed_0 = predictor.predict_articles(
        new_data_removed_0,
        details=["prediction_type",
                 "irrelevant_facts",
                 "irrelevant_fact_probabilities",
                 "relevant_fact_probabilities",
                 ],
        shlv=shlv)
    assert (new_data_removed_0 == dev_data)
    assert (answer_dict_removed_0 == answer_dict)
    assert (sp_dict_removed_0 == sp_dict)
    new_data_dicts = {}
    new_data_dicts[0] = new_data_removed_0
    new_answer_dicts = {}
    new_answer_dicts[0] = answer_dict_removed_0
    for k in range(1, num_removals):
        new_data_dicts[k] = remove_irrelevantly_labelled_facts_ordered(dev_data_dict,
                                                                       sp_dict,
                                                                       details_dict,
                                                                       num_removed=k)
        # Sort
        new_data_dicts[k] = [new_data_dicts[k][e] for e in sorted(new_data_dicts[k])]
        new_answer_dicts[k], _, _, _ = predictor.predict_articles(new_data_dicts[k],
                                                                  details=[
                                                                      "prediction_type",
                                                                      "irrelevant_facts",
                                                                      "irrelevant_fact_probabilities"],
                                                                  shlv=shlv)
    changed_answer_fractions_irr = {}
    for k in new_answer_dicts.keys():
        changed_answer_fractions_irr[k] = changed_answer_fraction(new_answer_dicts[0], new_answer_dicts[k])
    scores['c_{irr}(k)'] = changed_answer_fractions_irr
    farm_scores = {}
    for k in new_answer_dicts.keys():
        farm_scores[k] = scores['c_{rel}(k)'][k] / (1 + scores['c_{irr}(k)'][k])
    scores['FaRM(k)'] = farm_scores
    return scores


def changed_answer_fraction(answer_dict_after, answer_dict_before):
    # This function is used to calculate c_{rel}(k) and c_{irr}(k) defined in Equations 5 and 6 in the paper.
    flip_counter = 0
    total_counter = 0
    for _id in answer_dict_after.keys():
        before = answer_dict_before[_id]
        after = answer_dict_after[_id]
        if after != before:
            flip_counter += 1
        total_counter += 1
    changed_answer_fraction = flip_counter / total_counter
    return changed_answer_fraction


def remove_relevantly_labelled_facts_ordered(data_dict_format, sps_original, details_dict, num_removed=1,
                                             verbose=False):
    # sps_original are predicted, not ground truth
    new_data_dict = {}
    for id_ in sps_original.keys():
        # Copy everything except the context
        new_data_dict[id_] = data_dict_format[id_].copy()
        new_data_dict[id_].pop('context', None)
        sp_predictions = sps_original[id_]
        sp_probas = details_dict[id_]["relevant_fact_probabilities"]
        assert (len(sp_predictions) == len(sp_probas))
        # Now, we sort the predictions in decreasing order according to their probabilities
        sorted_idxs = np.argsort(sp_probas)[::-1]
        sorted_sp_predictions = [sp_predictions[idx] for idx in sorted_idxs]
        # Leave titles inside as they are no facts
        sorted_sp_predictions = [e for e in sorted_sp_predictions if e[1] != -1]
        if verbose:
            print(f"Relevancy ranking: {sorted_sp_predictions}")
        # Preprocess the relevant fact predictions into a nicer format
        sp_predictions_dict = {}
        for relevant_fact in sp_predictions:
            # Check if the title already is in the dict
            if not relevant_fact[0] in sp_predictions_dict:
                sp_predictions_dict[relevant_fact[0]] = []
            # Add the sentence index
            sp_predictions_dict[relevant_fact[0]].append(relevant_fact[1])
        # We directly determine which sentences out of the supporting facts we want to neglect
        sp_remove_idxs = {}
        for key in sp_predictions_dict.keys():
            sp_remove_idxs[key] = []
        for i in range(num_removed):
            if i == len(sorted_sp_predictions):
                if verbose:
                    print(f"Cant remove more facts because there are not enough "
                          f"relevant facts (removed {i}, wanted to "
                          f"remove {num_removed})")
                break
            remove_from_article = sorted_sp_predictions[i][0]
            remove_sent = sorted_sp_predictions[i][1]
            sp_remove_idxs[remove_from_article].append(remove_sent)
        # We step through the context and check if we encounter a fact that
        # the model labelled as relevant
        full_context = data_dict_format[id_]['context']
        reduced_context = []
        for article in full_context:
            title = article[0]
            # The structure is [[title1, [sent_1, sent_2, ...]], [title_2, ...]]
            reduced_article = [title, []]
            for sent_index, sent in enumerate(article[1]):
                # Predicted as relevant
                if title in sp_predictions_dict:
                    if sent_index in sp_predictions_dict[title]:
                        if verbose:
                            print("Relevant: " + sent)
                        if title in sp_remove_idxs:
                            if sent_index in sp_remove_idxs[title]:
                                if verbose:
                                    print("Removed: " + sent)
                            else:
                                reduced_article[1].append(sent)
                        else:
                            reduced_article[1].append(sent)
                    else:
                        reduced_article[1].append(sent)
                # Keep all articles in which no sentence was labelled as relevant
                else:
                    reduced_article[1].append(sent)

            reduced_context.append(reduced_article)
        new_data_dict[id_]['context'] = reduced_context
        if verbose:
            print(80 * '#')
    return new_data_dict


def remove_irrelevantly_labelled_facts_ordered(data_dict_format, sps_original, details_dict, num_removed=1,
                                               verbose=False):
    # sps_original are predicted, not ground truth
    new_data_dict = {}
    for id_ in sps_original.keys():
        # Copy everything except the context
        new_data_dict[id_] = data_dict_format[id_].copy()
        new_data_dict[id_].pop('context', None)
        irrelevant_facts = details_dict[id_]["irrelevant_facts"]
        irrelevant_fact_probabilities = details_dict[id_]["irrelevant_fact_probabilities"]
        assert (len(irrelevant_facts) == len(irrelevant_fact_probabilities))
        # Now, we sort the predictions in decreasing order according to their probabilities
        assert (len(irrelevant_fact_probabilities) == len(irrelevant_facts))
        sorted_idxs = np.argsort(irrelevant_fact_probabilities)[::-1]
        sorted_irrelevant_sp_predictions = [irrelevant_facts[idx] for idx in sorted_idxs]
        # Leave titles inside as they are no facts
        sorted_irrelevant_sp_predictions = [e for e in sorted_irrelevant_sp_predictions if e[1] != -1]
        if verbose:
            print(f"Relevancy ranking: {sorted_irrelevant_sp_predictions}")
        # Preprocess the relevant fact predictions into a nicer format
        irrelevant_sp_predictions_dict = {}
        for irrelevant_fact in irrelevant_facts:
            # Check if the title already is in the dict
            if not irrelevant_fact[0] in irrelevant_sp_predictions_dict:
                irrelevant_sp_predictions_dict[irrelevant_fact[0]] = []
            # Add the sentence index
            irrelevant_sp_predictions_dict[irrelevant_fact[0]].append(irrelevant_fact[1])
        # We directly determine which sentences out of the supporting facts we want to neglect
        sp_remove_idxs = {}
        for key in irrelevant_sp_predictions_dict.keys():
            sp_remove_idxs[key] = []
        for i in range(num_removed):
            if i == len(sorted_irrelevant_sp_predictions):
                if verbose:
                    print(f"Cant remove more facts because there are not enough "
                          f"relevant facts (removed {i}, wanted to "
                          f"remove {num_removed})")
                break
            remove_from_article = sorted_irrelevant_sp_predictions[i][0]
            remove_sent = sorted_irrelevant_sp_predictions[i][1]
            sp_remove_idxs[remove_from_article].append(remove_sent)
        # We step through the context and check if we encounter a fact that
        # the model labelled as relevant
        full_context = data_dict_format[id_]['context']
        reduced_context = []
        for article in full_context:
            title = article[0]
            # The structure is [[title1, [sent_1, sent_2, ...]], [title_2, ...]]
            reduced_article = [title, []]
            for sent_index, sent in enumerate(article[1]):
                # Predicted as relevant
                if title in irrelevant_sp_predictions_dict:
                    if sent_index in irrelevant_sp_predictions_dict[title]:
                        if verbose:
                            print("Irrelevant: " + sent)
                        if title in sp_remove_idxs:
                            if sent_index in sp_remove_idxs[title]:
                                if verbose:
                                    print("Removed: " + sent)
                            else:
                                reduced_article[1].append(sent)
                        else:
                            reduced_article[1].append(sent)
                    else:
                        reduced_article[1].append(sent)
                # Keep all articles in which no sentence was labelled as relevant
                else:
                    reduced_article[1].append(sent)
            reduced_context.append(reduced_article)
        new_data_dict[id_]['context'] = reduced_context
        if verbose:
            print(80 * '#')
    return new_data_dict
