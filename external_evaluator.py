import argparse
import os
from typing import List, Tuple, Dict
from sklearn.metrics import precision_recall_fscore_support as prfs
import json

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class ReEvaluator:
    def __init__(self, gt_path, pred_path):
        self.gt_entities = []
        self.gt_relations = []
        self.pred_entities = []
        self.pred_relations = []

        self.gt_entities, self.gt_relations, \
            self.pred_entities, self.pred_relations = self._read_gt_and_pred(gt_path, pred_path)

    def _convert_to_tuple(self, doc, dct):
        if 'start' in dct:
            return (dct['start'],
                    dct['end'],
                    dct['type'])
        return (doc['entities'][dct['head']]['start'],
                doc['entities'][dct['head']]['end'],
                doc['entities'][dct['head']]['type'],
                doc['entities'][dct['tail']]['start'],
                doc['entities'][dct['tail']]['end'],
                doc['entities'][dct['tail']]['type'],
                dct['type'])

    def _read_gt_and_pred(self, gt_path, pred_path):
        with open(gt_path, 'r') as f:
            gt = json.load(f)
        with open(pred_path, 'r') as f:
            pred = json.load(f)
        gt_entities = []
        gt_relations = []
        pred_entities = []
        pred_relations = []

        assert len(gt) == len(pred)
        for i in range(len(gt)):
            # Assume gt is in dictionary format and pred is in tuple format
            gt_entities.append([self._convert_to_tuple(gt[i], e) for e in gt[i]['entities']])
            gt_relations.append([self._convert_to_tuple(gt[i], e) for e in gt[i]['relations']])
            pred_entities.append([self._convert_to_tuple(pred[i], e) for e in pred[i]['entities']])
            pred_relations.append([self._convert_to_tuple(pred[i], e) for e in pred[i]['relations']])
        return gt_entities, gt_relations, pred_entities, pred_relations

    def _convert_by_setting(self, gt, pred, include_entity_types):
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if len(t) == 3:  # entity
                    c = [t[0], t[1], 'Entity']
                else:  # relation
                    c = [(t[0], t[1], 'Entity'),
                         (t[3], t[4], 'Entity'), t[6]]
            else:
                if len(t) == 3:
                    c = [t[0], t[1], t[2]]
                else:
                    c = [(t[0], t[1], t[2]),
                         (t[3], t[4], t[5]), t[6]]
            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred

    def compute_scores(self):
        print("Evaluation")

        print("")
        print("--- Entities (named entity recognition (NER)) ---")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self.gt_entities, self.pred_entities, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related entities are predicted correctly (entity type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(self.gt_relations, self.pred_relations, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True)

        print("")
        print("With named entity classification (NEC)")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(self.gt_relations, self.pred_relations, include_entity_types=True)
        rel_nec_eval = self._score(gt, pred, print_results=True)

        return ner_eval, rel_eval, rel_nec_eval

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if s in sample_gt:
                    t = s[2]
                    gt_flat.append(t)
                    types.add(t)
                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    pred_flat.append(t)
                    types.add(t)
                else:
                    pred_flat.append(0)

        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t for t in types]
        per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--pred_path', type=str)
    args = parser.parse_args()
    evaluator = ReEvaluator(args.gt_path, args.pred_path)
    evaluator.compute_scores()

