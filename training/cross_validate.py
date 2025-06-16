import os
import tempfile
import numpy as np
import pandas as pd

import train

from autosubsync import find_transform
from autosubsync import quality_of_fit
from autosubsync import model


def cv_split_by_file(data_meta, data_x):
    files = np.unique(data_meta.file_number)
    np.random.shuffle(files)

    n_train = int(round(len(files) * 0.5))
    train_files = files[:n_train]
    print(train_files)

    train_cols = data_meta.file_number.isin(train_files)
    test_cols = ~train_cols
    return data_meta[train_cols], data_x[train_cols, :], data_meta[test_cols], data_x[test_cols, :]


def validate_speech_detection(result_meta):
    print("---- speech detection accuracy ----")

    # Only aggregate numeric columns that make sense to average
    numeric_cols = ["label", "predicted_score", "predicted_label", "correct"]
    # Filter to only include columns that actually exist in the dataframe
    cols_to_agg = [col for col in numeric_cols if col in result_meta.columns]
    r = result_meta.groupby("file_number")[cols_to_agg].agg("mean")
    print(r)
    from sklearn.metrics import roc_auc_score

    print("AUC-ROC:", roc_auc_score(result_meta.label, result_meta.predicted_score))
    return r


def test_correct_sync(result_meta, bias=0):
    print("---- synchronization accuracy ----")

    results = []
    for file_number in np.unique(result_meta.file_number):
        part = result_meta[result_meta.file_number == file_number]
        skew, shift, quality = find_transform.find_transform_parameters(part.label, part.predicted_score, bias=bias)
        skew_error = skew != 1.0
        results.append([file_number, skew_error, shift, quality])
        print(results[-1])

    sync_results = pd.DataFrame(np.array(results), columns=["file_number", "skew_error", "shift_error", "quality"])
    print(sync_results)

    print("skew errors:", sync_results.skew_error.sum())
    print("shift RMSE:", np.sqrt(np.mean(sync_results.shift_error**2)))

    return sync_results


def test_quality_of_fit_mismatch(result_meta, bias=0):

    all_files = np.unique(result_meta.file_number)
    pairs = [(n1, n2) for n1 in all_files for n2 in all_files if n2 != n1]

    print("---- quality of fit (computing for %d mismatches) ----" % len(pairs))

    qualities = []

    for fn1, fn2 in pairs:
        labels0 = result_meta.label[result_meta.file_number == fn1]
        probs0 = result_meta.predicted_score[result_meta.file_number == fn2]
        l = min(len(labels0), len(probs0))

        labels = probs0 * 0
        labels[:l] = labels0[:l]

        for flip in [False, True]:
            if flip:
                probs = probs0[::-1]
            else:
                probs = probs0[::]

            skew, shift, quality = find_transform.find_transform_parameters(labels, probs, bias=bias)
            quality_error = quality >= quality_of_fit.threshold

            qualities.append(quality)
            print(quality)

    return np.array(qualities)


if __name__ == "__main__":

    data_x, data_meta = train.load_features()

    print("loaded training features of size", data_x.shape)
    n_folds = 4
    np.random.seed(1)

    sync_results = []

    for i in range(n_folds):
        print("### Cross-validation fold %d/%d" % (i + 1, n_folds))
        train_meta, train_x, test_meta, test_x = cv_split_by_file(data_meta, data_x)

        print("Training...", train_x.shape)
        trained_model = model.train(train_x, train_meta.label, train_meta, verbose=True)

        # save some memory
        del train_x
        del train_meta

        # test serialization
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "model.bin")
            print("testing serialization in temp file", tmp_file)
            model.save(trained_model, tmp_file)
            trained_model = model.load(tmp_file)

        print("Validating...")
        predicted_score = model.predict(trained_model, test_x, test_meta.file_number)
        result_meta = test_meta.assign(predicted_score=predicted_score)
        result_meta = result_meta.assign(predicted_label=np.round(predicted_score))
        result_meta = result_meta.assign(label=np.round(result_meta.label))
        result_meta = result_meta.assign(correct=result_meta.predicted_label == result_meta.label)

        bias = trained_model[1]
        r = validate_speech_detection(result_meta)
        sync_r = test_correct_sync(result_meta, bias)
        sync_results.append(sync_r.assign(speech_detection_accuracy=list(r.correct)))

    sync_results = pd.concat(sync_results)
    print(sync_results)
    print("skew errors:", sync_results.skew_error.sum())
    print("shift RMSE:", np.sqrt(np.mean(sync_results.shift_error**2)))
    print("shift max:", sync_results.shift_error.abs().max())
    print("shift bias, mean:", sync_results.shift_error.mean(), "median:", sync_results.shift_error.median())
    print("speech detection accuracy (mean of means):", sync_results.speech_detection_accuracy.mean())

    # save some more memory
    del data_x
    del data_meta

    correct_qualities = np.asarray(sync_results.quality)
    print("false negative quality errors:", np.sum(correct_qualities < quality_of_fit.threshold))

    print("### Quality of fit mismatch test (with last fold)")
    mismatch_qualities = test_quality_of_fit_mismatch(result_meta, bias)

    min_correct = np.min(correct_qualities)
    max_incorrect = np.max(mismatch_qualities)

    quality_margin = min_correct - max_incorrect

    print("estimated threshold:", (min_correct + max_incorrect) * 0.5)
    print("current threshold:", quality_of_fit.threshold)
    print("quality margin:", quality_margin)

    print("false positive quality errors:", np.sum(mismatch_qualities > quality_of_fit.threshold))
